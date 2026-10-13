import os
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'
import sys
import gc
import glob
import tempfile
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
import torch
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from torchvision import transforms
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.model_main import UrbanSeg
MOSAIC_DIR = 'data/mosaics'
OUTPUT_DIR = 'predictions'
CKPT_PATH = 'weights/last.ckpt'
CITIES = ['Beijing', 'Changchun', 'Changsha', 'Chengdu', 'Chongqing', 'Fuzhou', 'Guangzhou', 'Guiyang', 'Haikou', 'Hangzhou', 'Harbin', 'Hefei', 'Hohhot', 'Jinan', 'Kunming', 'Lanzhou', 'Lhasa', 'Nanchang', 'Nanjing', 'Nanning', 'Shanghai', 'Shenyang', 'Shijiazhuang', 'Taiyuan', 'Tianjin', 'Urumqi', 'Wuhan', "Xi'an", 'Xining', 'Yinchuan', 'Zhengzhou']
WINDOW_SIZE = 1536
TILE_SIZE = 1024
PADDING = 256
BATCH_SIZE = 16
USE_TTA = False
DO_PYRAMID = True
STRIP_H = 2048
CHUNK_H = 16384
CHUNK_W = 16384
CHUNK_MARGIN = 1536
SHP_PATH = ''
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PALETTE = {0: (0, 0, 0, 255), 1: (34, 139, 34, 255), 2: (0, 255, 0, 255), 3: (139, 69, 19, 255), 4: (128, 128, 128, 255), 5: (0, 0, 255, 255), 6: (255, 0, 0, 255), 7: (255, 255, 0, 255), 8: (255, 165, 0, 255)}
NC = 9

def load_model(ckpt_path, device):
    model = UrbanSeg(num_classes=9, img_size=WINDOW_SIZE).to(device)
    model.eval()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'state_dict' in ckpt:
        sd = {k.replace('model.', ''): v for k, v in ckpt['state_dict'].items() if k.startswith('model.')}
        model.load_state_dict(sd, strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)
    return model

@torch.no_grad()
def predict_batch(model, batch, device, use_tta=False):
    batch = batch.to(device)
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        _, logits = model(batch)
        if use_tta:
            l_hf = torch.flip(model(torch.flip(batch, (3,)))[1], (3,))
            l_vf = torch.flip(model(torch.flip(batch, (2,)))[1], (2,))
            logits = (logits + l_hf + l_vf) / 3.0
    return logits

def _gauss1d(L):
    half = (L - 1) / 2
    sigma = L / 4
    return np.exp(-0.5 * ((np.arange(L) - half) / sigma) ** 2)
_gm, _gs = (None, None)

def _blend_mask(sz):
    global _gm, _gs
    if _gm is None or _gs != sz:
        _gm = np.outer(_gauss1d(sz), _gauss1d(sz)).astype(np.float32)
        _gs = sz
    return _gm

def _build_tile_grid(H, W, tile_size):
    step = max(1, tile_size * 3 // 4)
    tiles = []
    for y in range(0, H, step):
        for x in range(0, W, step):
            ww = min(tile_size, W - x)
            wh = min(tile_size, H - y)
            tiles.append((x, y, ww, wh))
    return tiles

def _init_progress(progress_file, total):
    if progress_file:
        try:
            with open(progress_file, 'w') as pf:
                pf.write(f'TOTAL {total}\n')
        except OSError:
            pass

def _mark_block_done(progress_file, iy, ix):
    if progress_file:
        try:
            with open(progress_file, 'a') as pf:
                pf.write(f'{iy}_{ix}\n')
        except OSError:
            pass

def _load_done(progress_file):
    done = set()
    if progress_file and os.path.exists(progress_file):
        try:
            with open(progress_file) as pf:
                for line in pf:
                    s = line.strip()
                    if s and (not s.startswith('TOTAL')) and (not s.startswith('BOUNDS')):
                        done.add(s)
        except OSError:
            pass
    return done

def _progress_summary(progress_file):
    total = done = 0
    if progress_file and os.path.exists(progress_file):
        try:
            with open(progress_file) as pf:
                for line in pf:
                    s = line.strip()
                    if s.startswith('TOTAL'):
                        try:
                            total = int(s.split()[1])
                        except (IndexError, ValueError):
                            pass
                    elif s and (not s.startswith('BOUNDS')):
                        done += 1
        except OSError:
            pass
    return (done, total)

def _load_bounds(progress_file):
    if progress_file and os.path.exists(progress_file):
        try:
            with open(progress_file) as pf:
                for line in pf:
                    s = line.strip()
                    if s.startswith('BOUNDS'):
                        parts = s.split()
                        if len(parts) == 5:
                            return tuple((int(v) for v in parts[1:]))
        except OSError:
            pass
    return None

def _save_bounds(progress_file, bounds):
    if progress_file:
        try:
            with open(progress_file, 'a') as pf:
                pf.write(f"BOUNDS {' '.join((str(v) for v in bounds))}\n")
        except OSError:
            pass

def _flush_strip(model, tensors, infos, device, n, padding, s0, acc, wsum, mask, use_tta):
    stacked = torch.stack(tensors)
    logits = predict_batch(model, stacked, device, use_tta)
    probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
    for i in range(n):
        x, y, ww, wh = infos[i]
        p = probs[i, :, padding:padding + wh, padding:padding + ww]
        w = mask[:wh, :ww]
        ly0 = y - s0
        acc_start = max(0, ly0)
        acc_end = min(acc.shape[0], ly0 + wh)
        data_start = max(0, -ly0)
        clip_h = acc_end - acc_start
        if clip_h <= 0:
            continue
        acc_x0 = max(0, x)
        acc_x1 = min(acc.shape[1], x + ww)
        data_x0 = max(0, -x)
        clip_w = acc_x1 - acc_x0
        if clip_w <= 0:
            continue
        for c in range(p.shape[0]):
            acc[acc_start:acc_end, acc_x0:acc_x1, c] += p[c, data_start:data_start + clip_h, data_x0:data_x0 + clip_w] * w[data_start:data_start + clip_h, data_x0:data_x0 + clip_w]
        wsum[acc_start:acc_end, acc_x0:acc_x1] += w[data_start:data_start + clip_h, data_x0:data_x0 + clip_w]
    del stacked, logits, probs

def _infer_on_array(model, img_hwc, device, transform_fn, tile_size, window_size, padding, batch_size, use_tta, desc, tile_positions=None, show_pbar=True):
    H, W = (img_hwc.shape[0], img_hwc.shape[1])
    mask = _blend_mask(tile_size)
    if tile_positions is not None:
        strip_tiles = {}
        for x, y, ww, wh in tile_positions:
            first_s = y // STRIP_H * STRIP_H
            last_s = (y + wh - 1) // STRIP_H * STRIP_H
            for s in range(first_s, last_s + 1, STRIP_H):
                strip_tiles.setdefault(s, []).append((x, y, ww, wh))
        strips = sorted(strip_tiles.keys())
        total = len(tile_positions)
    else:
        step = max(1, tile_size * 3 // 4)
        ys = list(range(0, H, step))
        xs = list(range(0, W, step))
        strips = list(range(0, H, STRIP_H))
        strip_tiles = {s: [] for s in strips}
        for y in ys:
            for x in xs:
                ww = min(tile_size, W - x)
                wh = min(tile_size, H - y)
                first_s = y // STRIP_H * STRIP_H
                last_s = (y + wh - 1) // STRIP_H * STRIP_H
                for s in range(first_s, last_s + 1, STRIP_H):
                    if s in strip_tiles:
                        strip_tiles[s].append((x, y, ww, wh))
        total = len(ys) * len(xs)
    pred = np.zeros((H, W), dtype=np.uint8)
    pbar = tqdm(total=total, desc=desc, unit='tile', disable=not show_pbar)
    for s0 in strips:
        tiles = strip_tiles[s0]
        if not tiles:
            continue
        s1 = min(H, s0 + STRIP_H)
        sh = s1 - s0
        acc = np.zeros((sh, W, NC), dtype=np.float32)
        wsum = np.zeros((sh, W), dtype=np.float32)
        tensors, infos = ([], [])
        for x, y, ww, wh in tiles:
            rx, ry = (x - padding, y - padding)
            px0, px1 = (max(0, rx), min(W, rx + window_size))
            py0, py1 = (max(0, ry), min(H, ry + window_size))
            patch = np.zeros((window_size, window_size, 3), dtype=img_hwc.dtype)
            ph, pw = (py1 - py0, px1 - px0)
            patch[max(0, -ry):max(0, -ry) + ph, max(0, -rx):max(0, -rx) + pw] = img_hwc[py0:py1, px0:px1]
            if patch.max() == 0:
                ly0 = max(0, y - s0)
                ly1 = min(sh, y + wh - s0)
                if ly1 > ly0:
                    wsum[ly0:ly1, x:x + ww] += 1e-06
                pbar.update(1)
                continue
            tensors.append(transform_fn(patch))
            infos.append((x, y, ww, wh))
            if len(tensors) == batch_size:
                _flush_strip(model, tensors, infos, device, batch_size, padding, s0, acc, wsum, mask, use_tta)
                tensors, infos = ([], [])
        if tensors:
            _flush_strip(model, tensors, infos, device, len(tensors), padding, s0, acc, wsum, mask, use_tta)
        ok = wsum > 0
        sp = np.zeros((sh, W), dtype=np.uint8)
        sp[ok] = np.argmax(acc[ok] / wsum[ok, None], axis=-1).astype(np.uint8)
        pred[s0:s1] = sp
        del acc, wsum, sp, ok
        pbar.update(len(tiles))
    pbar.close()
    gc.collect()
    return pred

def compute_valid_bounds(src):
    H, W = (src.height, src.width)
    row_ok = np.zeros(H, dtype=bool)
    col_ok = np.zeros(W, dtype=bool)
    BS = 8192
    for r0 in range(0, H, BS):
        r1 = min(H, r0 + BS)
        for c0 in range(0, W, BS):
            c1 = min(W, c0 + BS)
            d = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0), boundless=True, fill_value=0)
            m = d != 0
            if m.any():
                row_ok[r0:r1] |= np.any(m, axis=1)
                col_ok[c0:c1] |= np.any(m, axis=0)
    if not row_ok.any():
        return None
    rows = np.where(row_ok)[0]
    cols = np.where(col_ok)[0]
    return (int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1)

def crop_to_bounds(tif_path, bounds, chunk=8192):
    r0, r1, c0, c1 = bounds
    with rasterio.open(tif_path) as ds:
        W, H = (ds.width, ds.height)
        if r0 == 0 and r1 == H and (c0 == 0) and (c1 == W):
            return
        win = Window(c0, r0, c1 - c0, r1 - r0)
        if win.height >= H - 2048 and win.width >= W - 2048:
            return
        prof = ds.profile.copy()
        prof.update(height=win.height, width=win.width, transform=ds.window_transform(win))
    tmp = tif_path + '.tmp'
    if os.path.exists(tmp):
        os.remove(tmp)
    with rasterio.open(tif_path) as src, rasterio.open(tmp, 'w', **prof) as dst:
        dst.write_colormap(1, PALETTE)
        for top in range(0, win.height, chunk):
            h = min(chunk, win.height - top)
            for left in range(0, win.width, chunk):
                w = min(chunk, win.width - left)
                out_win = Window(left, top, w, h)
                src_win = Window(win.col_off + left, win.row_off + top, w, h)
                arr = src.read(1, window=src_win)
                dst.write(arr, 1, window=out_win)
    os.replace(tmp, tif_path)

def build_pyramids(tif_path, factors=(2, 4, 8, 16, 32)):
    with rasterio.open(tif_path, 'r+') as ds:
        ds.build_overviews(list(factors), resampling=Resampling.nearest)
        ds.update_tags(ns='rio_overview', resampling='nearest')
        ds.write_colormap(1, PALETTE)

def mask_to_shp(tif_path, shp_path, chunk=8192, fill=0):
    import fiona
    from rasterio.features import geometry_mask
    with fiona.open(shp_path) as src:
        geoms = [feat['geometry'] for feat in src]
    with rasterio.open(tif_path, 'r+') as ds:
        H, W = (ds.height, ds.width)
        for top in range(0, H, chunk):
            h = min(chunk, H - top)
            for left in range(0, W, chunk):
                w = min(chunk, W - left)
                win = Window(left, top, w, h)
                mask = geometry_mask(geoms, out_shape=(h, w), transform=ds.window_transform(win), invert=False)
                data = ds.read(1, window=win)
                data[mask] = fill
                ds.write(data, 1, window=win)
        ds.write_colormap(1, PALETTE)

def infer_city(model, city, device, transform_fn, out_dir):
    src_path = os.path.join(MOSAIC_DIR, f'{city}.tif')
    out_path = os.path.join(out_dir, f'{city}.tif')
    done_flag = out_path + '.done'
    progress_file = out_path + '.progress'
    if not os.path.exists(src_path):
        print(f'[skip] {city}: mosaic 不存在')
        return
    if os.path.exists(done_flag):
        print(f'[skip] {city}: 已完成（断点续传）')
        return
    with rasterio.open(src_path) as src:
        H, W = (src.height, src.width)
        print(f'[{city}] mosaic {W}x{H}')
        cached_bounds = _load_bounds(progress_file)
        if cached_bounds is not None:
            valid_bounds = cached_bounds
            print(f'  使用缓存有效范围: rows [{valid_bounds[0]},{valid_bounds[1]}) cols [{valid_bounds[2]},{valid_bounds[3]})')
        else:
            valid_bounds = compute_valid_bounds(src)
            if valid_bounds is None:
                print('  [warn] 未检测到有效像元，按全图处理')
                valid_bounds = (0, H, 0, W)
            print(f'  有效范围: rows [{valid_bounds[0]},{valid_bounds[1]}) cols [{valid_bounds[2]},{valid_bounds[3]})')
        r0, r1, c0, c1 = valid_bounds
        shp_geom = None
        if SHP_PATH and os.path.exists(SHP_PATH):
            import fiona
            from shapely.geometry import shape
            from shapely.prepared import prep
            with fiona.open(SHP_PATH) as shp_src:
                feat = next(iter(shp_src))
                shp_geom = prep(shape(feat['geometry']))
            print(f'  SHP mask: {SHP_PATH}')
        all_tiles = _build_tile_grid(H, W, TILE_SIZE)
        print(f'  全局 tile grid: {len(all_tiles)} tiles')
        ys = list(range(0, H, CHUNK_H))
        xs = list(range(0, W, CHUNK_W))
        total = len(ys) * len(xs)
        block_tiles = {}
        for gx, gy, gw, gh in all_tiles:
            first_by = max(0, gy // CHUNK_H)
            last_by = min(len(ys) - 1, (gy + gh - 1) // CHUNK_H)
            first_bx = max(0, gx // CHUNK_W)
            last_bx = min(len(xs) - 1, (gx + gw - 1) // CHUNK_W)
            for by in range(first_by, last_by + 1):
                for bx in range(first_bx, last_bx + 1):
                    key = (by, bx)
                    if key not in block_tiles:
                        block_tiles[key] = []
                    block_tiles[key].append((gx, gy, gw, gh))
        meta = src.meta.copy()
        meta.update(driver='GTiff', count=1, dtype='uint8', compress='lzw', nodata=0, bigtiff='YES', tiled=True)
        done_set = _load_done(progress_file)
        if done_set:
            print(f'  断点续传: {len(done_set)} block(s) 已完成，跳过')
        candidates = []
        for iy, cy in enumerate(ys):
            ch = min(CHUNK_H, H - cy)
            for ix, cx in enumerate(xs):
                cw = min(CHUNK_W, W - cx)
                if f'{iy}_{ix}' in done_set:
                    continue
                gt = block_tiles.get((iy, ix), [])
                if not gt:
                    continue
                if not (cy < r1 and cy + ch > r0 and (cx < c1) and (cx + cw > c0)):
                    continue
                if shp_geom is not None:
                    from shapely.geometry import box as sbox
                    x0, y0 = src.xy(cy, cx)
                    x1, y1 = src.xy(min(cy + ch, H - 1), min(cx + cw, W - 1))
                    if not shp_geom.intersects(sbox(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))):
                        continue
                candidates.append((iy, ix, cy, cx, ch, cw))
        skipped = total - len(candidates)
        print(f'  待处理块: {len(candidates)}（跳过 {skipped}: 断点/无tile/有效范围外/SHP外）')
        out_exists = os.path.exists(out_path)
        progress_exists = os.path.exists(progress_file)
        reinitialized = False
        if out_exists and (not progress_exists):
            os.remove(out_path)
            out_exists = False
        if out_exists:
            try:
                dst = rasterio.open(out_path, 'r+')
            except Exception:
                print('  输出文件损坏，重建')
                os.remove(out_path)
                done_set = set()
                _init_progress(progress_file, total)
                reinitialized = True
                dst = rasterio.open(out_path, 'w', **meta)
        else:
            _init_progress(progress_file, total)
            dst = rasterio.open(out_path, 'w', **meta)
        if cached_bounds is None or reinitialized:
            _save_bounds(progress_file, valid_bounds)
        failed_blocks = []
        with dst:
            dst.write_colormap(1, PALETTE)
            pbar_blocks = tqdm(total=total, initial=skipped, desc=f'  {city} blocks')
            import threading, queue
            q = queue.Queue(maxsize=2)

            def _reader():
                rsrc = rasterio.open(src_path)
                try:
                    for iy, ix, cy, cx, ch, cw in candidates:
                        try:
                            ry0 = max(0, cy - CHUNK_MARGIN)
                            ry1 = min(H, cy + ch + CHUNK_MARGIN)
                            rx0 = max(0, cx - CHUNK_MARGIN)
                            rx1 = min(W, cx + cw + CHUNK_MARGIN)
                            ctiles = [(gx - rx0, gy - ry0, gw, gh) for gx, gy, gw, gh in block_tiles[iy, ix]]
                            img = rsrc.read((1, 2, 3), window=Window(rx0, ry0, rx1 - rx0, ry1 - ry0), boundless=True, fill_value=0)
                            ihwc = np.transpose(img, (1, 2, 0)).copy()
                            del img
                            if ihwc.max() == 0:
                                q.put((iy, ix, cy, cx, ch, cw, None, None, 'empty'))
                            else:
                                q.put((iy, ix, cy, cx, ch, cw, ihwc, ctiles, 'ok'))
                        except Exception as e:
                            print(f'  [WARN] block {iy}_{ix} 读取失败: {e}', flush=True)
                            q.put((iy, ix, cy, cx, ch, cw, None, None, 'error'))
                finally:
                    rsrc.close()
                    q.put(None)
            threading.Thread(target=_reader, daemon=True).start()
            while True:
                item = q.get()
                if item is None:
                    break
                iy, ix, cy, cx, ch, cw, ihwc, ctiles, status = item
                if status == 'error':
                    pbar_blocks.update(1)
                    failed_blocks.append((iy, ix))
                    continue
                if status == 'empty':
                    pbar_blocks.update(1)
                    _mark_block_done(progress_file, iy, ix)
                    continue
                pbar_blocks.set_postfix_str(f'[{cy}:{cy + ch},{cx}:{cx + cw}]')
                pred_chunk = _infer_on_array(model, ihwc, device, transform_fn, TILE_SIZE, WINDOW_SIZE, PADDING, BATCH_SIZE, USE_TTA, desc=f'  [{cy}:{cy + ch},{cx}:{cx + cw}]', tile_positions=ctiles, show_pbar=False)
                del ihwc, ctiles
                ry0 = max(0, cy - CHUNK_MARGIN)
                rx0 = max(0, cx - CHUNK_MARGIN)
                ct = cy - ry0
                cb = ct + ch
                cl = cx - rx0
                cr = cl + cw
                safe_pred = pred_chunk[ct:cb, cl:cr]
                dst.write(safe_pred, 1, window=Window(cx, cy, cw, ch))
                del pred_chunk, safe_pred
                pbar_blocks.update(1)
                _mark_block_done(progress_file, iy, ix)
                gc.collect()
            pbar_blocks.close()
    if failed_blocks:
        print(f'[FAIL] {city} 有 {len(failed_blocks)} 个块读取失败，未写完成标记，可重跑续传: {failed_blocks[:10]}')
        return
    crop_to_bounds(out_path, valid_bounds)
    if SHP_PATH and os.path.exists(SHP_PATH):
        mask_to_shp(out_path, SHP_PATH)
    if DO_PYRAMID:
        build_pyramids(out_path)
    if os.path.exists(progress_file):
        os.remove(progress_file)
    open(done_flag, 'w').close()
    print(f'[{city}] 完成 -> {out_path}')

def main():
    torch.backends.cudnn.benchmark = True
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}（物理卡 2）')
    model = load_model(CKPT_PATH, device)
    transform_fn = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cities = CITIES
    print(f'推理城市: {len(cities)} 个')
    for city in cities:
        infer_city(model, city, device, transform_fn, OUTPUT_DIR)
    print('全部完成')
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
if __name__ == '__main__':
    main()
