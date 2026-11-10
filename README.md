# UrbanSegNet / CHRUC-2024

Urban land-cover segmentation with DINOv3 ViT-L/16, LoRA, three MRDB blocks and confidence-filtered pseudo-labels.

## Classes

1 Trees · 2 Grass/Shrub · 3 Soil · 4 OISAs · 5 Water · 6 Buildings · 7 Roads · 8 Agriculture.
Code 0 denotes background/no data; 255 is the cross-entropy ignore code.

## Setup

Python 3.10; PyTorch 2.1.1 with CUDA 11.8. See `environment.json` for the reported server environment.

```bash
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Install Triton and any selective-scan CUDA extension for your platform separately. The reported environment uses Triton 3.3.0; Windows needs a compatible build.

## Weights

Place the final `last.ckpt` in `weights/`. SHA256:
`a2f3a84766fb2cd59bf8e2c3d374cfec2aa4dfa0b33d50216b51359e203452f0`.
Download [`last.ckpt` and its accompanying licences](https://github.com/hmbbsett/UrbanSegNet/releases/tag/chruc-2024-v1.0). The complete DINO-derived checkpoint is distributed under the accompanying DINOv3 License.

For training from scratch, obtain `dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth` from [Meta DINOv3](https://github.com/facebookresearch/dinov3) and place it in `weights/`.
It is not redistributed here. For inference, the complete final checkpoint supplies the backbone and LoRA parameters.

## Train

Set the training data path in `config.py`, then run from the repository root:

```bash
python train.py --name abl_4city --cities Beijing,Guangzhou,Wuhan,Hohhot --encoder dinov3 --tau 0.4 --lambda_aux 0.4
```

Training uses 1024×1024 tiles, eight epochs, batch size 4 and gradient accumulation 8.

## Evaluate the final maps

Extract the 2,000-mask reference archive. Its root contains `manifest.csv` and `GT/`. Use the final city GeoTIFFs named `<City>.tif` and run:

```bash
pip install numpy rasterio
python tools/evaluate_product.py --maps-dir data/31citylabel --reference-dir data/reference
```

The evaluator reads each 100 × 100 reference window from the final city map and pools the pixel counts across patches. Unclassified product pixels count as errors.

| Scope | Patches | Reference pixels | OA (%) | mPA (%) | mUA (%) | mF1 (%) | mIoU (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All 31 cities | 2,000 | 20,000,000 | 84.04 | 82.62 | 84.61 | 83.14 | 71.93 |
| 30-city comparison | 1,973 | 19,730,000 | 84.06 | 82.43 | 84.66 | 83.04 | 71.81 |

`tools/inference_validate.py` and its lists under `evaluation/lists/` are retained for historical single-window experiments.

## Map cities

Set `MOSAIC_DIR`, `OUTPUT_DIR`, `CKPT_PATH` and `CITIES` in `inference.py`, then run:

```bash
python inference.py
```

Production uses 1536×1536 inputs, 1024×1024 output tiles, stride 768 and Gaussian blending. Output is a single-band UInt8 GeoTIFF.

## Data and licenses

The CHRUC-2024 dataset consists of the 31 city GeoTIFFs and 2,000 manually labelled reference masks under CC BY 4.0.
Source imagery is obtained separately from [Jilin-1](https://www.jl1mall.com/).
See `LICENSE`, `NOTICE` and `licenses/` for code and third-party terms.
