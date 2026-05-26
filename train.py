import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassF1Score
from torch.optim.lr_scheduler import CosineAnnealingLR
from config import update_config, _C as cfg
from dataset import train_aug, val_aug, UrbanSegDataset
from models.model_main import UrbanSeg

# Class names for evaluation output
CLASS_NAMES = [
    'Background', 'Tree', 'Grass', 'Soil',
    'OISA', 'Water', 'Building', 'Road', 'Crop'
]


class SegmentationLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.5, class_weights=None,label_smoothing=0.0):
        super(SegmentationLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=255,
            label_smoothing=label_smoothing
        )

    def _dice_loss(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)

        valid_mask = (targets != 255).float().unsqueeze(1)

        targets_safe = targets.clone()
        targets_safe[targets == 255] = 0

        target_one_hot = F.one_hot(targets_safe, num_classes=self.num_classes).permute(0, 3, 1, 2).type_as(inputs)

        target_one_hot = target_one_hot * valid_mask

        smooth = 1e-5
        dims = (0, 2, 3)
        intersection = torch.sum(inputs * target_one_hot, dim=dims)
        cardinality = torch.sum(inputs + target_one_hot, dim=dims)
        dice_score = (2. * intersection + smooth) / (cardinality + smooth)
        return 1. - torch.mean(dice_score)

    def forward(self, inputs, targets):
        ce = self.ce_loss(inputs, targets)
        dice = self._dice_loss(inputs, targets)
        return self.alpha * ce + (1 - self.alpha) * dice


class UrbanSegLightningSystem(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        self.num_classes = 9

        self.model = UrbanSeg(num_classes=self.num_classes, img_size=config.DATA.IMG_SIZE)

        trainable_params = 0
        total_params = 0

        target_keys = ['w_a', 'w_b', 'decoder', 'mamba', 'head', 'aux', 'norm', 'bias']

        for name, param in self.model.named_parameters():
            total_params += param.numel()

            param.requires_grad = False

            is_lora_part = any(k in name for k in target_keys)

            if is_lora_part :
                param.requires_grad = True

            if param.requires_grad:
                trainable_params += param.numel()

        class_weights = torch.tensor([
            1.0,  # Background
            2.5,  # Tree
            3.0,  # Grass
            3.0,  # Soil
            2.0,  # OISA
            1.0,  # Water
            1.5,  # Building
            2.0,  # Road
            3.0  # Crop
        ]).float()

        self.criterion = SegmentationLoss(
            num_classes=self.num_classes,
            alpha=0.5,
            class_weights=class_weights,
            label_smoothing=config.MODEL.LABEL_SMOOTHING
        )

        self.train_iou_metric = MulticlassJaccardIndex(num_classes=self.num_classes, ignore_index=255, average=None)
        self.train_f1_metric = MulticlassF1Score(num_classes=self.num_classes, ignore_index=255, average=None)
        self.val_iou_metric = MulticlassJaccardIndex(num_classes=self.num_classes, ignore_index=255, average=None)
        self.val_f1_metric = MulticlassF1Score(num_classes=self.num_classes, ignore_index=255, average=None)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg'].long()
        logits_aux, logits_main = self(img)
        refined_mask = mask.clone()


        probs_aux = F.softmax(logits_aux, dim=1)
        conf_aux, preds_aux = torch.max(probs_aux, dim=1)
        clean_threshold = 0.4  # confidence threshold for self-cleansing
        potential_dirty_mask = (preds_aux !=  mask) & (conf_aux > clean_threshold)
        refined_mask[potential_dirty_mask] = 255
        self.log('cleaned_rate', potential_dirty_mask.float().mean(), prog_bar=True, on_step=False, on_epoch=True)

        loss_aux = self.criterion(logits_aux, mask)
        loss_main = self.criterion(logits_main, refined_mask)
        loss = loss_main + 0.4 * loss_aux

        preds_main = torch.argmax(logits_main, dim=1)
        self.train_iou_metric.update(preds_main, mask)
        self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg'].long()
        _, logits_main = self(img)
        loss = self.criterion(logits_main, mask)
        preds = logits_main.argmax(dim=1)
        self.val_iou_metric.update(preds, mask)
        self.val_f1_metric.update(preds, mask)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.train_iou_metric.reset()
        self.train_f1_metric.reset()

    def on_validation_epoch_end(self):
        ious = self.val_iou_metric.compute()
        print(f"\n[Epoch {self.current_epoch} VAL Metrics]")
        for i in range(len(ious)):
            if i < len(CLASS_NAMES):
                print(f"{CLASS_NAMES[i]:<10} | {ious[i] * 100:6.2f}")

        if len(ious) > 1:
            mean_iou = ious[1:].mean() * 100
            print(f"Mean IoU (wo/BG): {mean_iou:.2f}")
            self.log('val_mIoU', mean_iou, prog_bar=True)
        else:
            self.log('val_mIoU', 0.0, prog_bar=True)

        self.val_iou_metric.reset()
        self.val_f1_metric.reset()

    def configure_optimizers(self):
        lora_params = []
        decoder_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad: continue

            if 'w_a' in name or 'w_b' in name:
                lora_params.append(param)

            elif 'decoder' in name or 'head' in name or 'mamba' in name or 'aux' in name:
                decoder_params.append(param)

        print(
            f"Optimizer: LoRA({len(lora_params)}) | Decoder({len(decoder_params)})")

        optimizer = optim.AdamW([
            {'params': decoder_params, 'lr': 1e-4},
            {'params': lora_params, 'lr': 2e-4}
        ], weight_decay=0.01)

        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.TRAIN.EPOCHS, eta_min=1e-7)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    torch.set_float32_matmul_precision('medium')
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.py')
    parser.add_argument('-r', '--resume', type=str, default=None)
    args = parser.parse_args()

    update_config(cfg, args.config)
    pl.seed_everything(cfg.SEED)

    train_loader = torch.utils.data.DataLoader(
        UrbanSegDataset(data_root=cfg.DATA.DATA_PATH, mode='train', img_size=cfg.DATA.IMG_SIZE, transform=train_aug),
        batch_size=cfg.DATA.BATCH_SIZE, shuffle=True, persistent_workers=True,
        num_workers=cfg.DATA.num_workers, drop_last=True, pin_memory=True,prefetch_factor=2)

    val_loader = torch.utils.data.DataLoader(
        UrbanSegDataset(data_root=cfg.DATA.DATA_PATH, mode='val', img_size=cfg.DATA.IMG_SIZE, transform=val_aug),
        batch_size=cfg.DATA.BATCH_SIZE, shuffle=False, persistent_workers=True,
        num_workers=cfg.DATA.num_workers, pin_memory=True)

    system = UrbanSegLightningSystem(cfg)

    checkpoint_callback = ModelCheckpoint(
        filename='{epoch}-{step}',
        monitor='val_mIoU',
        mode='max',
        save_top_k=1,
        save_last=True
    )

    trainer = pl.Trainer(
        max_epochs=cfg.TRAIN.EPOCHS,
        accelerator='gpu',
        devices=[0],
        callbacks=[checkpoint_callback],
        logger=CSVLogger('logs', name='UrbanSeg_DINOv3_Turbo'),
        precision="bf16-mixed",
        accumulate_grad_batches=8,
        gradient_clip_val=1.0
    )

    if args.resume:
        trainer.fit(system, train_loader, val_loader, ckpt_path=args.resume)
    else:
        trainer.fit(system, train_loader, val_loader)

if __name__ == "__main__":
    main()