import os
import sys
import tempfile
_gpu_arg = None
if '--gpu' in sys.argv[1:]:
    _gi = sys.argv[1:].index('--gpu')
    _gpu_arg = sys.argv[2 + _gi] if 2 + _gi < len(sys.argv) else '0'
else:
    for _a in sys.argv[1:]:
        if _a.startswith('--gpu='):
            _gpu_arg = _a.split('=', 1)[1]
            break
if _gpu_arg is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(_gpu_arg)
elif 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import csv
import io as _io
import time as _time
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassF1Score
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from config import update_config, _C as cfg
from dataset import train_aug, val_aug, UrbanSegDataset
from models.model_main import UrbanSeg
from models.cmp_models import build_model
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
CLASS_NAMES = ['Background', 'Tree', 'Grass', 'Soil', 'OISA', 'Water', 'Building', 'Road', 'Crop']
PALETTE_RGB = [[0, 0, 0], [34, 139, 34], [0, 255, 0], [139, 69, 19], [128, 128, 128], [0, 0, 255], [255, 0, 0], [255, 255, 0], [255, 165, 0]]

class SegmentationLoss(nn.Module):

    def __init__(self, num_classes, alpha=0.5, class_weights=None, label_smoothing=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255, label_smoothing=label_smoothing)

    def _dice_loss(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        valid_mask = (targets != 255).float().unsqueeze(1)
        targets_safe = targets.clone()
        targets_safe[targets == 255] = 0
        target_one_hot = F.one_hot(targets_safe, num_classes=self.num_classes).permute(0, 3, 1, 2).type_as(inputs)
        target_one_hot = target_one_hot * valid_mask
        smooth = 1e-05
        dims = (0, 2, 3)
        intersection = torch.sum(inputs * target_one_hot, dim=dims)
        cardinality = torch.sum(inputs + target_one_hot, dim=dims)
        dice_score = (2.0 * intersection + smooth) / (cardinality + smooth)
        return 1.0 - torch.mean(dice_score)

    def forward(self, inputs, targets):
        ce = self.ce_loss(inputs, targets)
        dice = self._dice_loss(inputs, targets)
        return self.alpha * ce + (1 - self.alpha) * dice

def mask_to_rgb(mask):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i, c in enumerate(PALETTE_RGB):
        rgb[mask == i] = c
    return rgb

class UrbanSegAblationSystem(pl.LightningModule):

    def __init__(self, config, tau=0.4, no_mamba=False, lambda_aux=0.4, encoder_type='dinov3', arch='urbanges', warmup=0, loss_abort=50.0, no_block=False, probe_only=False, train_ups=False):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.num_classes = 9
        self.tau = tau
        self.lambda_aux = lambda_aux
        self.encoder_type = encoder_type
        self.arch = arch
        self.warmup = warmup
        self.loss_abort = loss_abort
        if arch in (None, 'urbanges'):
            self.model = UrbanSeg(num_classes=self.num_classes, img_size=config.DATA.IMG_SIZE, encoder_type=encoder_type, probe_only=probe_only)
            target_keys = ['w_a', 'w_b', 'decoder', 'mamba', 'head', 'aux']
            if encoder_type != 'dinov3':
                target_keys = target_keys + ['encoder']
            if probe_only:
                target_keys = ['w_a', 'w_b', 'aux']
            if train_ups:
                target_keys = target_keys + ['up_16_to_8', 'up_8_to_4']
            for name, param in self.model.named_parameters():
                param.requires_grad = False
                if any((k in name for k in target_keys)):
                    param.requires_grad = True
            if probe_only:
                tb = sum((p.numel() for p in self.model.parameters() if p.requires_grad))
                print(f'[probe_only] 只有编码器（DINOv3 + LoRA）+ 1x1 卷积分类器；decoder_in / MRDB / 上采样 / head 均不参与前向。可训练 {tb:,}')
        else:
            self.model = build_model(arch, num_classes=self.num_classes, img_size=config.DATA.IMG_SIZE)
            for param in self.model.parameters():
                param.requires_grad = True
        if no_block:
            if arch not in (None, 'urbanges'):
                raise ValueError('--no_block 只对 urbanges 有效，不能与 --arch 对比模型同用')
            if no_mamba:
                raise ValueError('--no_block 与 --no_mamba 互斥：--no_block 已经去掉了整个 MRDB（含 SS2D 分支），不要再叠 --no_mamba')
            n_removed = sum((p.numel() for p in self.model.mamba_layers.parameters()))
            self.model.mamba_layers = nn.ModuleList()
            trainable = sum((p.numel() for p in self.model.parameters() if p.requires_grad))
            print(f'[no_block] 已删除整个 MRDB（3 个 MobileMambaDecoderBlock）：移除参数 {n_removed:,}；剩余可训练 {trainable:,}')
            self.no_block = True
        else:
            self.no_block = False
        if no_mamba:
            if arch not in (None, 'urbanges'):
                raise ValueError('--no_mamba 只对 urbanges 有效，不能与 --arch 对比模型同用')
            import models.model_main as mm

            def forward_no_mamba(self_block, x):
                shortcut = x
                x_g, x_l, x_id = torch.split(x, [self_block.g_dim, self_block.l_dim, self_block.i_dim], dim=1)
                x_g_out = x_g
                x_l_out = self_block.local_conv(x_l)
                x_id_out = x_id
                x_mixed = torch.cat([x_g_out, x_l_out, x_id_out], dim=1)
                x_mixed = self_block.mrffi_proj(x_mixed)
                x_mixed = self_block.mrffi_norm(x_mixed)
                x = shortcut + x_mixed
                x = x + self_block.ffn(self_block.norm_ffn(x))
                return x
            mm.MobileMambaDecoderBlock.forward = forward_no_mamba
            self.no_mamba = True
        else:
            self.no_mamba = False
        class_weights = torch.tensor([1.0, 2.5, 3.0, 3.0, 2.0, 1.0, 1.5, 2.0, 3.0]).float().cuda()
        self.criterion = SegmentationLoss(num_classes=self.num_classes, alpha=0.5, class_weights=class_weights, label_smoothing=config.MODEL.LABEL_SMOOTHING)
        self.train_iou_metric = MulticlassJaccardIndex(num_classes=9, ignore_index=255, average=None)
        self.val_iou_metric = MulticlassJaccardIndex(num_classes=9, ignore_index=255, average=None)
        self.val_f1_metric = MulticlassF1Score(num_classes=9, ignore_index=255, average=None)
        self.filtered_by_class = torch.zeros(9, dtype=torch.float64)
        self.total_by_class = torch.zeros(9, dtype=torch.float64)
        self._val_samples = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, mask = (batch['img'], batch['gt_semantic_seg'].long())
        logits_aux, logits_main = self(img)
        refined_mask = mask.clone()
        probs_aux = F.softmax(logits_aux, dim=1)
        conf_aux, preds_aux = torch.max(probs_aux, dim=1)
        potential_dirty = (preds_aux != mask) & (conf_aux > self.tau)
        refined_mask[potential_dirty] = 255
        self.log('cleaned_rate', potential_dirty.float().mean(), prog_bar=True, on_step=False, on_epoch=True)
        with torch.no_grad():
            dirty_cls = mask[potential_dirty]
            valid_cls = mask[mask < 255]
            if dirty_cls.numel() > 0:
                self.filtered_by_class += torch.bincount(dirty_cls, minlength=9).double().cpu()
            if valid_cls.numel() > 0:
                self.total_by_class += torch.bincount(valid_cls, minlength=9).double().cpu()
        loss_aux = self.criterion(logits_aux, mask)
        loss_main = self.criterion(logits_main, refined_mask)
        loss = loss_main + self.lambda_aux * loss_aux
        if not torch.isfinite(loss) or loss.item() > self.loss_abort:
            raise RuntimeError(f'[发散] train_loss = {loss.item()}（epoch {self.current_epoch}, step {batch_idx}）超过阈值 {self.loss_abort}，已中止训练。\n  体检已有 ckpt：python tools\\diag_predict.py\n  最常见原因：全量训练没加 warmup，重跑时加 --warmup 500。')
        preds_main = torch.argmax(logits_main, dim=1)
        self.train_iou_metric.update(preds_main, mask)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def on_train_epoch_start(self):
        self._epoch_t0 = _time.time()

    def on_train_epoch_end(self):
        out_dir = self.trainer.logger.log_dir if self.trainer.logger else 'logs/ablation'
        os.makedirs(out_dir, exist_ok=True)
        fpath = os.path.join(out_dir, 'filter_stats.csv')
        ep = self.current_epoch
        row = {'epoch': ep}
        dt = _time.time() - getattr(self, '_epoch_t0', _time.time())
        n_steps = max(1, self.trainer.global_step - getattr(self, '_step0', 0))
        self._step0 = self.trainer.global_step
        import config as _cfg_mod
        _append_csv(os.path.join(self.trainer.logger.log_dir, 'speed.csv'), {'epoch': ep, 'epoch_time_s': round(dt, 1), 'steps': n_steps, 'it_per_s': round(n_steps / dt, 3), 'samples_per_s': round(n_steps * cfg.DATA.BATCH_SIZE * 8 / dt, 3)})
        for i in range(9):
            f = self.filtered_by_class[i].item()
            t = self.total_by_class[i].item()
            row[f'cls{i}_filtered'] = int(f)
            row[f'cls{i}_total'] = int(t)
            row[f'cls{i}_rate'] = round(f / t, 6) if t > 0 else ''
        write_header = not os.path.exists(fpath)
        with _io.open(fpath, 'a', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)
        self.filtered_by_class.zero_()
        self.total_by_class.zero_()
        self.train_iou_metric.reset()

    def validation_step(self, batch, batch_idx):
        img, mask = (batch['img'], batch['gt_semantic_seg'].long())
        _, logits_main = self(img)
        loss = self.criterion(logits_main, mask)
        preds = logits_main.argmax(dim=1)
        self.val_iou_metric.update(preds, mask)
        self.val_f1_metric.update(preds, mask)
        self.log('val_loss', loss, prog_bar=True)
        if len(self._val_samples) < 3 and batch_idx < 5:
            self._val_samples.append((img[0].detach().cpu(), preds[0].detach().cpu(), mask[0].detach().cpu()))
        return loss

    def on_validation_epoch_end(self):
        ious = self.val_iou_metric.compute()
        print(f'\n[Epoch {self.current_epoch} VAL Metrics]')
        for i in range(len(ious)):
            if i < len(CLASS_NAMES):
                print(f'{CLASS_NAMES[i]:<10} | {ious[i] * 100:6.2f}')
        if len(ious) > 1:
            mean_iou = ious[1:].mean() * 100
            print(f'Mean IoU (wo/BG): {mean_iou:.2f}')
            self.log('val_mIoU', mean_iou, prog_bar=True)
        self.val_iou_metric.reset()
        self.val_f1_metric.reset()
        out_dir = self.trainer.logger.log_dir if self.trainer.logger else 'logs/ablation'
        samp_dir = os.path.join(out_dir, 'samples')
        os.makedirs(samp_dir, exist_ok=True)
        for si, (im, pr, mk) in enumerate(self._val_samples):
            im_np = im.numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            im_np = np.clip((im_np * std + mean) * 255, 0, 255).astype(np.uint8)
            pr_np = pr.numpy().astype(np.int64)
            mk_np = mk.numpy().astype(np.int64)
            mk_np[mk_np == 255] = 0
            panel = np.concatenate([im_np, mask_to_rgb(pr_np), mask_to_rgb(mk_np)], axis=1)
            from PIL import Image
            Image.fromarray(panel).save(os.path.join(samp_dir, f'epoch{self.current_epoch:02d}_s{si}.png'))
            Image.fromarray(mask_to_rgb(pr_np)).save(os.path.join(samp_dir, f'epoch{self.current_epoch:02d}_s{si}_pred.png'))
        self._val_samples = []

    def configure_optimizers(self):
        lora_params, decoder_params = ([], [])
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'w_a' in name or 'w_b' in name:
                lora_params.append(param)
            else:
                decoder_params.append(param)
        print(f'[Optimizer] arch={self.arch} | LoRA({len(lora_params)}) | Other({len(decoder_params)})')
        groups = [{'params': decoder_params, 'lr': 0.0001}]
        if lora_params:
            groups.append({'params': lora_params, 'lr': 0.0002})
        optimizer = optim.AdamW(groups, weight_decay=0.01)
        if self.warmup > 0:
            total = max(self.warmup + 1, int(self.trainer.estimated_stepping_batches))
            warm = LinearLR(optimizer, start_factor=0.01, total_iters=self.warmup)
            cos = CosineAnnealingLR(optimizer, T_max=max(1, total - self.warmup), eta_min=1e-07)
            sched = SequentialLR(optimizer, [warm, cos], milestones=[self.warmup])
            print(f'[Scheduler] 线性 warmup {self.warmup} 步 + 余弦退火（共 {total} 步）')
            return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': sched, 'interval': 'step'}}
        print(f'[Scheduler] 余弦退火（按 epoch，T_max={self.config.TRAIN.EPOCHS}，无 warmup）')
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.TRAIN.EPOCHS, eta_min=1e-07)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

def plot_curves(log_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    mpath = os.path.join(log_dir, 'metrics.csv')
    if not os.path.exists(mpath):
        return
    rows = list(csv.DictReader(_io.open(mpath, encoding='utf-8')))
    epochs, miou, cleaned, tloss = ([], [], [], [])
    for r in rows:
        ep = r.get('epoch')
        if ep is None or ep == '':
            continue
        try:
            ep = int(float(ep))
        except ValueError:
            continue
        if r.get('val_mIoU') not in ('', None):
            epochs.append(ep)
            miou.append(float(r['val_mIoU']))
        if r.get('cleaned_rate') not in ('', None):
            cleaned.append((ep, float(r['cleaned_rate'])))
        if r.get('train_loss') not in ('', None):
            tloss.append((ep, float(r['train_loss'])))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    if miou:
        axes[0].plot(epochs, miou, 'o-')
        axes[0].set_title('val_mIoU (%)')
        axes[0].set_xlabel('epoch')
    if cleaned:
        epc = [c[0] for c in cleaned]
        vc = [c[1] * 100 for c in cleaned]
        axes[1].plot(epc, vc, 'o-')
        axes[1].set_title('cleaned_rate (%)')
        axes[1].set_xlabel('epoch')
    if tloss:
        ept = [c[0] for c in tloss]
        vt = [c[1] for c in tloss]
        axes[2].plot(ept, vt, 'o-')
        axes[2].set_title('train_loss')
        axes[2].set_xlabel('epoch')
    fig.tight_layout()
    fig.savefig(os.path.join(log_dir, 'curves.png'), dpi=150)
    print('curves saved:', os.path.join(log_dir, 'curves.png'))

def _append_csv(path, row):
    new = not os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)

class SaveLastEachEpoch(pl.Callback):

    def __init__(self, dirpath):
        super().__init__()
        self.dirpath = dirpath

    def on_train_epoch_end(self, trainer, pl_module):
        os.makedirs(self.dirpath, exist_ok=True)
        path = os.path.join(self.dirpath, 'last.ckpt')
        trainer.save_checkpoint(path)
        print(f'[ckpt] saved {path}', flush=True)

class CkptAtEpochs(pl.Callback):

    def __init__(self, epochs):
        super().__init__()
        self.epochs = set(epochs)

    def on_train_epoch_end(self, trainer, pl_module):
        ep = trainer.current_epoch
        if ep in self.epochs:
            path = os.path.join(trainer.logger.log_dir, 'checkpoints', f'epoch{ep:02d}.ckpt')
            trainer.save_checkpoint(path)
            print(f'[ckpt] saved {path}')

def main():
    torch.set_float32_matmul_precision('medium')
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.py')
    parser.add_argument('--tau', type=float, default=0.4)
    parser.add_argument('--lambda_aux', type=float, default=0.4)
    parser.add_argument('--no_mamba', action='store_true')
    parser.add_argument('--no_block', action='store_true')
    parser.add_argument('--probe_only', action='store_true')
    parser.add_argument('--train_ups', action='store_true')
    parser.add_argument('--encoder', type=str, default='dinov3', choices=['dinov3', 'resnet18', 'resnet50'])
    parser.add_argument('--arch', type=str, default='urbanges')
    parser.add_argument('--subsample', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--accum', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--loss_abort', type=float, default=50.0)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--ckpt_epochs', type=str, default='')
    parser.add_argument('--no_val', action='store_true')
    parser.add_argument('--name', type=str, default='abl')
    parser.add_argument('--log_root', type=str, default='logs/ablation')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--cities', type=str, default=None)
    parser.add_argument('--limit_train_batches', type=int, default=None)
    parser.add_argument('--limit_val_batches', type=int, default=None)
    args = parser.parse_args()
    if args.arch not in (None, 'urbanges'):
        parser.error('--arch 只支持 urbanges')
    if args.epochs:
        cfg.defrost()
        cfg.TRAIN.EPOCHS = args.epochs
        cfg.freeze()
    update_config(cfg, args.config)
    if args.seed is not None:
        cfg.defrost()
        cfg.SEED = args.seed
        cfg.freeze()
    pl.seed_everything(cfg.SEED)
    train_ds = UrbanSegDataset(data_root=cfg.DATA.DATA_PATH, mode='train', img_size=cfg.DATA.IMG_SIZE, transform=train_aug)
    if args.cities:
        city_set = {c.strip() for c in args.cities.split(',')}
        keep = [i for i in range(len(train_ds)) if train_ds.image_paths[i].stem.split('_')[0] in city_set]
        train_ds = torch.utils.data.Subset(train_ds, keep)
        print(f'cities filter: {len(keep)} tiles kept')
    if args.subsample < 1.0:
        n = int(len(train_ds) * args.subsample)
        train_ds = torch.utils.data.Subset(train_ds, np.random.default_rng(0).choice(len(train_ds), n, replace=False))
        print(f'subsample: {len(train_ds)} tiles')
    val_ds = UrbanSegDataset(data_root=cfg.DATA.DATA_PATH, mode='val', img_size=cfg.DATA.IMG_SIZE, transform=val_aug)
    bs = args.batch_size if args.batch_size > 0 else cfg.DATA.BATCH_SIZE
    nw = cfg.DATA.num_workers
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=True, persistent_workers=True, num_workers=nw, drop_last=True, pin_memory=True, prefetch_factor=2)
    val_loader = None
    if not args.no_val:
        val_ds = UrbanSegDataset(data_root=cfg.DATA.DATA_PATH, mode='val', img_size=cfg.DATA.IMG_SIZE, transform=val_aug)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=bs, shuffle=False, persistent_workers=True, num_workers=nw, pin_memory=True, prefetch_factor=2)
    system = UrbanSegAblationSystem(cfg, tau=args.tau, no_mamba=args.no_mamba, lambda_aux=args.lambda_aux, encoder_type=args.encoder, arch=args.arch, warmup=args.warmup, loss_abort=args.loss_abort, no_block=args.no_block, probe_only=args.probe_only, train_ups=args.train_ups)
    log_dir = os.path.join(args.log_root, args.name)
    os.makedirs(log_dir, exist_ok=True)
    logger = CSVLogger(save_dir=os.path.dirname(log_dir), name=args.name)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    if args.no_val:
        print('[no_val] 训练时不跑验证，只保存 last.ckpt（每 epoch 一次）')
        cbs = [SaveLastEachEpoch(ckpt_dir)]
    else:
        cbs = [ModelCheckpoint(dirpath=ckpt_dir, filename='{epoch}-{step}', monitor='val_mIoU', mode='max', save_top_k=1, save_last=True)]
    if args.ckpt_epochs:
        cbs.append(CkptAtEpochs([int(e) for e in args.ckpt_epochs.split(',') if e.strip()]))
    trainer = pl.Trainer(max_epochs=cfg.TRAIN.EPOCHS, accelerator='gpu', devices=1, logger=logger, callbacks=cbs, limit_train_batches=args.limit_train_batches, limit_val_batches=0 if args.no_val else args.limit_val_batches, log_every_n_steps=50, enable_checkpointing=not args.no_val, num_sanity_val_steps=0, accumulate_grad_batches=args.accum, precision='bf16-mixed', gradient_clip_val=1.0)
    sig = f"cities={args.cities or 'ALL'};encoder={args.encoder};no_mamba={args.no_mamba};tau={args.tau};arch={args.arch}"
    if args.warmup > 0:
        sig += f';warmup={args.warmup}'
    if args.no_block:
        sig += ';no_block=True'
    if args.probe_only:
        sig += ';probe_only=True'
    if args.train_ups:
        sig += ';train_ups=True'
    if args.lambda_aux != 0.4:
        sig += f';lambda_aux={args.lambda_aux}'
    if args.seed is not None:
        sig += f';seed={args.seed}'
    exp_dir = os.path.join(args.log_root, args.name)
    config_file = os.path.join(exp_dir, 'ablation_config.txt')
    resume_path = args.resume
    if not resume_path:
        cands = []
        if os.path.isdir(exp_dir):
            top = os.path.join(exp_dir, 'checkpoints', 'last.ckpt')
            if os.path.exists(top):
                cands.append(top)
            versions = sorted([d for d in os.listdir(exp_dir) if d.startswith('version_')])
            for v in reversed(versions):
                c = os.path.join(exp_dir, v, 'checkpoints', 'last.ckpt')
                if os.path.exists(c):
                    cands.append(c)
                    break
        if cands and os.path.exists(config_file):
            old_sig = open(config_file, encoding='utf-8').read().strip()
            if old_sig == sig:
                resume_path = max(cands, key=os.path.getmtime)
            else:
                print(f'[resume] 配置与上次不同，从头开始：\n  旧: {old_sig}\n  新: {sig}')
        elif cands:
            print('[resume] 旧实验无配置签名，从头开始（避免错误续训）')
    os.makedirs(exp_dir, exist_ok=True)
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(sig + '\n')
    if resume_path:
        print(f'[resume] 检测到同配置检查点，将从 {resume_path} 继续训练')
    else:
        print('[train] 从头开始训练')
    trainer.fit(system, train_loader, val_loader, ckpt_path=resume_path or None)
    real_log_dir = logger.log_dir if logger.log_dir else log_dir
    print('\n=== filter_stats.csv（每类过滤统计）===')
    fs = os.path.join(real_log_dir, 'filter_stats.csv')
    if os.path.exists(fs):
        print(_io.open(fs, encoding='utf-8').read())
    plot_curves(real_log_dir)
    print(f'输出目录: {real_log_dir}')
if __name__ == '__main__':
    main()
