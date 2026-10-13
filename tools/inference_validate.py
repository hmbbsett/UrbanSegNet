import os
import sys
import tempfile
import argparse
import numpy as np
import rasterio
for _i, _a in enumerate(sys.argv):
    if _a == '--gpu' and _i + 1 < len(sys.argv):
        os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[_i + 1]
import torch
from torchvision import transforms
from rasterio.windows import Window
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cmp_models import build_model
CLASSES = ['Trees', 'Grass', 'Soil', 'OISAs', 'Water', 'Buildings', 'Roads', 'Agriculture']
MOSAIC_DIR = 'data/mosaics'
FD = 'data/reference'
GY_DIR = 'data/reference_guiyang'
WINDOW = 1536
CKPT_SIZE = 1024
CTX = (WINDOW - 256) // 2
TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def build_val_list(kept_txt, include_guiyang=True, gt_source='crop100', gt_dir=None):
    base = os.path.abspath(gt_dir) if gt_dir else FD
    items = []
    if os.path.exists(kept_txt):
        for line in open(kept_txt, encoding='utf-8'):
            n = line.strip()
            if not n:
                continue
            city = n.split('_')[0]
            try:
                y, x = (int(n.split('_')[2]), int(n.split('_')[3].replace('.tif', '')))
            except (IndexError, ValueError):
                continue
            p256 = os.path.join(base, 'gt_masks_256', n)
            p100 = os.path.join(base, 'crop_100_masks', n)
            if gt_source == 'gt256':
                gt, mode = (p256, 'full256')
            elif gt_source == 'auto':
                if os.path.exists(p256):
                    gt, mode = (p256, 'full256')
                else:
                    gt, mode = (p100, 'crop100')
            else:
                gt, mode = (p100, 'crop100')
            if os.path.exists(gt):
                items.append((city, y, x, gt, mode))
        if not items:
            sys.exit(f'❌ 名单里一个 GT 都没找到。\n   名单 : {kept_txt}\n   GT 根: {base}\n   贵阳请加 --gt_dir "<...>\\extracted_256x256_Guiyang"')
    if include_guiyang:
        cand = os.path.join(os.path.dirname(os.path.abspath(kept_txt)), 'validation_kept_guiyang_final.txt')
        if os.path.exists(cand):
            gy_names = [x.strip() for x in open(cand, encoding='utf-8') if x.strip()]
            print(f'[guiyang] 名单 {cand}  共 {len(gy_names)} 张')
        else:
            gy_names = sorted(os.listdir(os.path.join(GY_DIR, 'images'))) if os.path.isdir(os.path.join(GY_DIR, 'images')) else []
            print(f'[guiyang] ⚠️ 未找到筛选名单，回退为目录遍历（{len(gy_names)} 张） —— 这会让分母变成 291 张，慎用')
        for n in gy_names:
            city = n.split('_')[0]
            y, x = (int(n.split('_')[2]), int(n.split('_')[3].replace('.tif', '')))
            gt = os.path.join(GY_DIR, 'crop_100_masks', n)
            if os.path.exists(gt):
                items.append((city, y, x, gt, 'crop100'))
    return items

def read_gt(path, mode):
    with rasterio.open(path) as src:
        g = src.read(1)
    return g

def infer_center_batch(model, items, device, batch=4):
    BATCH = batch
    results = []
    for bi in range(0, len(items), BATCH):
        chunk = items[bi:bi + BATCH]
        tensors, metas = ([], [])
        for city, y, x, gt_p, mode in chunk:
            try:
                mp = os.path.join(MOSAIC_DIR, f'{city}.tif')
                with rasterio.open(mp) as src:
                    col0, row0 = (x - CTX, y - CTX)
                    w = Window(col0, row0, WINDOW, WINDOW)
                    patch = src.read((1, 2, 3), window=w, boundless=True, fill_value=0)
                patch = np.transpose(patch, (1, 2, 0))
                tensors.append(TRANSFORM(patch))
                metas.append((city, y, x, gt_p, mode))
            except Exception as e:
                print('  skip', city, y, x, e)
        if not tensors:
            continue
        x_t = torch.stack(tensors).to(device)
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                _, main = model(x_t)
        prob = torch.softmax(main, dim=1)
        preds = prob.argmax(dim=1).cpu().numpy().astype(np.uint8)
        for meta, pred in zip(metas, preds):
            results.append((pred[CTX:CTX + 256, CTX:CTX + 256],) + meta)
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--name', default='exp')
    ap.add_argument('--kept_txt', default='evaluation/lists/validation_30cities.txt')
    ap.add_argument('--no_guiyang', action='store_true')
    ap.add_argument('--save_maps', default='')
    ap.add_argument('--gt_dir', default='')
    ap.add_argument('--gt_source', default='crop100', choices=['crop100', 'gt256', 'auto'])
    ap.add_argument('--out_dir', default='evaluation/results')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--encoder', type=str, default='dinov3', choices=['dinov3', 'resnet18', 'resnet50'])
    ap.add_argument('--arch', type=str, default='urbanges')
    ap.add_argument('--no_block', action='store_true')
    ap.add_argument('--probe_only', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--zero_lora', action='store_true')
    ap.add_argument('--upsample', choices=['bilinear', 'nearest'], default='bilinear')
    args = ap.parse_args()
    import models.model_main as _mm
    _mm.UPSAMPLE_MODE = args.upsample
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True
    model = build_model(args.arch, num_classes=9, img_size=CKPT_SIZE, encoder_type=args.encoder, probe_only=args.probe_only).to(device).eval()
    if args.upsample == 'nearest':
        n_up = 0
        for _m in model.modules():
            if isinstance(_m, torch.nn.Upsample):
                _m.mode = 'nearest'
                n_up += 1
        print(f'[upsample] nearest —— 已把 {n_up} 个 nn.Upsample 换成最近邻，最终输出也用最近邻')
    if args.probe_only:
        print('[probe_only] 只有编码器（DINOv3+LoRA）+ 1x1 卷积分类器，解码器不参与前向')
    if args.no_block:
        model.mamba_layers = torch.nn.ModuleList()
        print('[no_block] 已删除整个 MRDB，按无解码块的 ckpt 加载')
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    sd = {k[len('model.'):]: v for k, v in ck['state_dict'].items() if k.startswith('model.')}
    model.load_state_dict(sd, strict=True)
    print(f'[loaded] {args.ckpt}')
    if args.zero_lora:
        n = 0
        with torch.no_grad():
            for pname, p in model.named_parameters():
                if 'w_b' in pname:
                    p.zero_()
                    n += 1
        print(f'[zero_lora] 已把 {n} 个 w_b 清零 —— LoRA 增量归零，等价于未经适配的原始 DINOv3 主干')
    items = build_val_list(args.kept_txt, include_guiyang=not args.no_guiyang, gt_source=args.gt_source, gt_dir=args.gt_dir or None)
    if args.limit:
        items = items[:args.limit]
    print(f'[val list] {len(items)} patches')
    os.makedirs(args.out_dir, exist_ok=True)
    results = infer_center_batch(model, items, device, batch=args.batch)
    CM = np.zeros((8, 8), dtype=np.int64)
    n_ok = 0
    if args.save_maps:
        os.makedirs(args.save_maps, exist_ok=True)
    for i, (pred, city, y, x, gt_p, mode) in enumerate(results):
        try:
            gt = read_gt(gt_p, mode)
        except Exception as e:
            print('  skip', city, y, x, e)
            continue
        if mode == 'crop100':
            pred = pred[78:178, 78:178]
        if args.save_maps:
            sub = os.path.join(args.save_maps, city)
            os.makedirs(sub, exist_ok=True)
            op = os.path.join(sub, os.path.basename(gt_p))
            try:
                with rasterio.open(gt_p) as s:
                    prof = s.profile.copy()
                prof.update(count=1, dtype='uint8', compress='lzw', nodata=0)
                prof.pop('photometric', None)
                with rasterio.open(op, 'w', **prof) as d:
                    d.write(pred.astype(np.uint8), 1)
            except Exception as e:
                print('  save_maps 失败', op, e)
        valid = (gt >= 1) & (gt <= 8) & (pred >= 1) & (pred <= 8)
        if valid.sum() == 0:
            continue
        for a, b in zip(gt[valid], pred[valid]):
            CM[a - 1, b - 1] += 1
        n_ok += 1
        if (i + 1) % 100 == 0:
            print(f'  {i + 1}/{len(items)}')
    oa = np.trace(CM) / CM.sum() if CM.sum() else 0
    print(f'\n=== {args.name} ===')
    print(f'OA = {oa * 100:.2f}%   (patches={n_ok}, pixels={CM.sum()})')
    print('混淆矩阵（行=GT, 列=Pred）:')
    print('        ' + ''.join((f'{c:>7}' for c in CLASSES)))
    for g in range(8):
        print(f'{CLASSES[g]:>7} ' + ''.join((f'{CM[g, p]:>7}' for p in range(8))) + f'  GT={CM[g].sum()}')
    rows = []
    for i, c in enumerate(CLASSES):
        tp, gt, pr = (CM[i, i], CM[i].sum(), CM[:, i].sum())
        pa = tp / gt if gt else 0
        ua = tp / pr if pr else 0
        f1 = 2 * pa * ua / (pa + ua) if pa + ua else 0
        iou = tp / (gt + pr - tp) if gt + pr - tp else 0
        rows.append(f'{c},{pa * 100:.2f},{ua * 100:.2f},{f1 * 100:.2f},{iou * 100:.2f}')
        print(f'  {c:>8}: PA={pa * 100:6.2f}  UA={ua * 100:6.2f}  F1={f1 * 100:6.2f}  IoU={iou * 100:6.2f}')
    with open(os.path.join(args.out_dir, f'{args.name}_oa.csv'), 'w', encoding='utf-8') as f:
        f.write(f'name,OA,patches,pixels\n{args.name},{oa * 100:.2f},{n_ok},{CM.sum()}\n')
    with open(os.path.join(args.out_dir, f'{args.name}_perclass.csv'), 'w', encoding='utf-8') as f:
        f.write('class,PA,UA,F1,IoU\n' + '\n'.join(rows) + '\n')
    np.savetxt(os.path.join(args.out_dir, f'{args.name}_confusion.csv'), CM, delimiter=',', header=','.join(CLASSES), comments='', fmt='%d')
    print(f'\n[saved] {args.out_dir}/{args.name}_*.csv')
if __name__ == '__main__':
    main()
