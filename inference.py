import os
import argparse
import torch
import numpy as np
import rasterio
from rasterio.windows import Window
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path

from models.model_main import UrbanSeg

WINDOW_SIZE = 1536
TILE_SIZE = 1024
PADDING = 256

BATCH_SIZE = 4

PALETTE = {
    0: (0, 0, 0, 255),
    1: (34, 139, 34, 255),
    2: (0, 255, 0, 255),
    3: (139, 69, 19, 255),
    4: (128, 128, 128, 255),
    5: (0, 0, 255, 255),
    6: (255, 0, 0, 255),
    7: (255, 255, 0, 255),
    8: (255, 165, 0, 255)
}


def load_model(ckpt_path, device):
    model = UrbanSeg(num_classes=9, img_size=WINDOW_SIZE).to(device)
    model.eval()
    checkpoint = torch.load(ckpt_path, map_location=device)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
        model.load_state_dict(state_dict, strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)
    return model


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='UrbanSeg large-TIFF inference with sliding window.')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing input GeoTIFF (.tif) files.')
    parser.add_argument('--output-dir', type=str, default='./predicted_masks',
                        help='Directory to save predicted masks (default: ./predicted_masks).')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to the model checkpoint (.ckpt) file.')
    parser.add_argument('--window-size', type=int, default=1536)
    parser.add_argument('--tile-size', type=int, default=1024)
    parser.add_argument('--padding', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()

    global WINDOW_SIZE, TILE_SIZE, PADDING, BATCH_SIZE
    WINDOW_SIZE = args.window_size
    TILE_SIZE = args.tile_size
    PADDING = args.padding
    BATCH_SIZE = args.batch_size

    torch.backends.cudnn.benchmark = True

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.ckpt, device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tif_files = list(Path(args.input_dir).glob("*.tif"))
    if not tif_files:
        print(f"No .tif files found in {args.input_dir}")
        return

    for tif_path in tif_files:
        out_path = Path(args.output_dir) / f"{tif_path.stem}_pred.tif"

        with rasterio.open(tif_path) as src:
            meta = src.meta.copy()
            height, width = src.height, src.width

            meta.update({'count': 1, 'dtype': rasterio.uint8, 'compress': 'lzw', 'nodata': 0})

            with rasterio.open(out_path, 'w', **meta) as dst:
                dst.write_colormap(1, PALETTE)

                y_steps = list(range(0, height, TILE_SIZE))
                x_steps = list(range(0, width, TILE_SIZE))
                total_steps = len(y_steps) * len(x_steps)

                batch_tensors = []
                batch_infos = []

                with tqdm(total=total_steps, desc=f"Inference ({tif_path.name})") as pbar:
                    for y in y_steps:
                        for x in x_steps:
                            write_w = min(TILE_SIZE, width - x)
                            write_h = min(TILE_SIZE, height - y)
                            window_write = Window(x, y, write_w, write_h)

                            read_window = Window(x - PADDING, y - PADDING, WINDOW_SIZE, WINDOW_SIZE)
                            patch_chw = src.read((1, 2, 3), window=read_window, boundless=True, fill_value=0)
                            patch_hwc = np.transpose(patch_chw, (1, 2, 0))

                            if patch_hwc.max() == 0:
                                dst.write(np.zeros((write_h, write_w), dtype=rasterio.uint8), 1, window=window_write)
                                pbar.update(1)
                                continue

                            input_tensor = transform(patch_hwc)
                            batch_tensors.append(input_tensor)
                            batch_infos.append((window_write, write_h, write_w))

                            if len(batch_tensors) == BATCH_SIZE:
                                batch_input = torch.stack(batch_tensors).to(device)

                                with torch.autocast(device_type='cuda', dtype=torch.float16):
                                    _, logits = model(batch_input)

                                preds = torch.argmax(logits, dim=1).cpu().numpy().astype(rasterio.uint8)

                                for i in range(BATCH_SIZE):
                                    win, h, w = batch_infos[i]
                                    valid_pred = preds[i, PADDING: PADDING + h, PADDING: PADDING + w]
                                    dst.write(valid_pred, 1, window=win)
                                    pbar.update(1)

                                batch_tensors = []
                                batch_infos = []
                    if len(batch_tensors) > 0:
                        batch_input = torch.stack(batch_tensors).to(device)

                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            _, logits = model(batch_input)

                        preds = torch.argmax(logits, dim=1).cpu().numpy().astype(rasterio.uint8)

                        for i in range(len(preds)):
                            win, h, w = batch_infos[i]
                            valid_pred = preds[i, PADDING: PADDING + h, PADDING: PADDING + w]
                            dst.write(valid_pred, 1, window=win)
                            pbar.update(1)

        print(f"Done: {tif_path.name}")


if __name__ == "__main__":
    main()
