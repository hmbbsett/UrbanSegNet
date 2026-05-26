# UrbanSeg: Urban Remote Sensing Semantic Segmentation

UrbanSeg is a semantic segmentation model for high-resolution urban satellite imagery. It classifies each pixel into one of 9 land-cover categories using a hybrid architecture combining **DINOv3 (Vision Transformer)** with **Mamba State Space Models (SSM)** and a lightweight CNN decoder.

## Land-Cover Classes

| Index | Class      | Color          |
|-------|------------|----------------|
| 0     | Background | Black          |
| 1     | Tree       | Forest Green   |
| 2     | Grass      | Green          |
| 3     | Soil       | Brown          |
| 4     | OISA*      | Gray           |
| 5     | Water      | Blue           |
| 6     | Building   | Red            |
| 7     | Road       | Yellow         |
| 8     | Crop       | Orange         |

*OISA = Other Impervious Surface Area

## Architecture

```
Input RGB Image
    │
    ▼
DINOv3 ViT-L/16 (frozen backbone + LoRA adapters)
    │
    ▼
3× Mamba Decoder Blocks (SS2D + Multi-Kernel Depthwise Conv + FFN)
    │
    ▼
Upsampling + Segmentation Head → Pixel-wise predictions
```

- **Encoder**: DINOv3 ViT-L/16 pretrained on satellite imagery (SAT493M), fine-tuned with LoRA
- **Decoder**: Lightweight MobileMamba blocks with 2D Selective Scan SSM
- **Loss**: Combined Cross-Entropy + Dice loss with class weights and self-cleansing

## Directory Structure

```
UrbanSeg/
├── config.py              # YACS-based configuration
├── dataset.py             # Data loading & augmentation
├── train.py               # Training script (PyTorch Lightning)
├── inference.py           # Large-TIFF inference with sliding window
├── models/
│   ├── encoder.py         # DINOv3 encoder with LoRA
│   ├── mamba_core.py      # 2D Mamba SSM (SS2D)
│   ├── csm_triton.py      # Cross-scan/merge (Triton kernels)
│   ├── model_main.py      # Full UrbanSeg model
│   └── ...
├── dinov3/                # Vendored DINOv3 library (Meta)
├── tools/
│   ├── cfg.py             # Config utilities
│   └── metric.py          # Evaluation metrics
└── weights/               # Pretrained weights (download separately)
```

## Installation

```bash
pip install -r requirements.txt
```

For GPU-accelerated selective scan kernels (optional but recommended):

```bash
pip install triton
# selective-scan-cuda may need to be built from source
```

## Data Preparation

Organize your data as follows:

```
<data_root>/
├── images/
│   ├── train/
│   │   ├── image_001.png
│   │   └── ...
│   └── val/
│       └── ...
└── labels/
    ├── train/
    │   ├── image_001.png
    │   └── ...
    └── val/
        └── ...
```

- Images: RGB, any resolution (resized to `IMG_SIZE` during training)
- Labels: Single-channel PNG with integer class indices (0-8, 255 for ignore)

Update `config.py` or use `-c` to point to your data root:

```python
_C.DATA.DATA_PATH = './data/Provincial_Capital'
```

## Pretrained Weights

The DINOv3 encoder requires pretrained weights. Download the **SAT493M** checkpoint:

```bash
mkdir -p weights
# Download dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth into weights/
```

The download URL can be found in the [DINOv3 repository](https://github.com/facebookresearch/dinov3).

## Training

Edit `config.py` to set your data path and hyperparameters, then run:

```bash
python train.py
```

To use a custom config file:

```bash
python train.py -c path/to/your_config.py
```

Resume from a checkpoint:

```bash
python train.py -r logs/UrbanSeg_DINOv3_Turbo/version_0/checkpoints/last.ckpt
```

### Key Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATA.DATA_PATH` | `./data/Provincial_Capital` | Path to dataset |
| `DATA.IMG_SIZE` | 1024 | Input image size (square) |
| `DATA.BATCH_SIZE` | 4 | Batch size per GPU |
| `TRAIN.EPOCHS` | 8 | Number of training epochs |
| `TRAIN.BASE_LR` | 2e-4 | Base learning rate |

## Inference

Run inference on large GeoTIFF files using a sliding window approach:

```bash
python inference.py \
    --input-dir /path/to/tiff/files \
    --output-dir /path/to/output \
    --ckpt path/to/checkpoint.ckpt
```

Optional arguments:

```bash
--window-size 1536   # Context window size (default: 1536)
--tile-size 1024     # Output tile size (default: 1024)
--padding 256        # Padding to avoid edge artifacts (default: 256)
--batch-size 4       # Batch size for inference (default: 4)
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

The `dinov3/` directory contains third-party code from Meta Platforms, Inc., which is governed by its own license. See the copyright headers in those files.

## Citation

If you use UrbanSeg in your research, please cite:

```bibtex
@software{urbanseg,
  title = {UrbanSeg: Urban Remote Sensing Semantic Segmentation},
  year = {2025},
  url = {https://github.com/<your-username>/UrbanSeg}
}
```
