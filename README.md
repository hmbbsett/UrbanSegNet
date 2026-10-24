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

## Evaluate

Set `MOSAIC_DIR` in `tools/inference_validate.py`. Put fixed 100×100 reference labels in `<reference_root>/crop_100_masks/`.

```bash
python tools/inference_validate.py --ckpt weights/last.ckpt --name chruc_30cities --kept_txt evaluation/lists/validation_30cities.txt --gt_dir data/reference --gt_source crop100 --no_guiyang --batch 8
python tools/inference_validate.py --ckpt weights/last.ckpt --name chruc_guiyang --kept_txt evaluation/lists/validation_guiyang.txt --gt_dir data/reference_guiyang --gt_source crop100 --no_guiyang --batch 8
```

The 2090-patch, 30-city paper result is OA 84.06% and pooled eight-class mIoU 71.62%.
Evaluation uses a 1536×1536 context window and the central 100×100 reference area.

## Map cities

Set `MOSAIC_DIR`, `OUTPUT_DIR`, `CKPT_PATH` and `CITIES` in `inference.py`, then run:

```bash
python inference.py
```

Production uses 1536×1536 inputs, 1024×1024 output tiles, stride 768 and Gaussian blending. Output is a single-band UInt8 GeoTIFF.

## Data and licenses

The 31-city product and 2380 manual 100×100 reference masks are available from [Figshare](https://doi.org/10.6084/m9.figshare.34764945) under CC BY 4.0.
Source imagery is obtained separately from [Jilin-1](https://www.jl1mall.com/).
See `LICENSE`, `NOTICE` and `licenses/` for code and third-party terms.
