import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from typing import Optional, Callable, Tuple, Dict
import torch
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform(height: int, width: int):

    train_transform = [
        A.Resize(height=height, width=width),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]
    return A.Compose(train_transform)


def get_val_transform(height: int, width: int):
    val_transform = [
        A.Resize(height=height, width=width),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]
    return A.Compose(val_transform)


def train_aug(img, mask, img_size):
    if not isinstance(img, np.ndarray): img = np.array(img)
    if not isinstance(mask, np.ndarray): mask = np.array(mask)

    aug = get_train_transform(height=img_size[0], width=img_size[1])(image=img, mask=mask)
    img, mask = aug['image'], aug['mask']

    mask = mask.long()

    return img, mask


def val_aug(img, mask, img_size):
    target_height, target_width = img_size[0], img_size[1]

    if img.shape[0] != target_height or img.shape[1] != target_width:
        img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

    aug = get_val_transform(height=target_height, width=target_width)(image=img, mask=mask)
    img, mask = aug['image'], aug['mask']

    mask = mask.long()
    return img, mask


class UrbanSegDataset(Dataset):
    def __init__(self,
                 data_root: str,
                 mode: str = 'train',
                 transform: Optional[Callable] = None,
                 img_size: Tuple[int, int] = (1024, 1024),
                 img_dir: str = 'images',
                 mask_dir: str = 'labels'):

        self.data_root = Path(data_root)
        self.mode = mode
        self.transform = transform

        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
        else:
            self.img_size = img_size

        current_img_folder = self.data_root / img_dir / self.mode
        current_mask_folder = self.data_root / mask_dir / self.mode

        if not current_img_folder.is_dir():
            if (self.data_root / self.mode / img_dir).is_dir():
                current_img_folder = self.data_root / self.mode / img_dir
                current_mask_folder = self.data_root / self.mode / mask_dir
            else:
                raise FileNotFoundError(f"Image directory not found: {current_img_folder}. "
                                        f"Expected structure: <data_root>/{{images,labels}}/{{train,val}}/")

        print(f"[{mode}] Scanning files: {current_img_folder}")
        self.image_paths = sorted(list(current_img_folder.glob("*.png")))
        self.mask_paths = [current_mask_folder / p.name for p in self.image_paths]

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in {current_img_folder}")

        print(f"[{mode}] Found {len(self.image_paths)} images.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_from_disk(self, index):
        img_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        img = cv2.imread(str(img_path))
        if img is None: raise IOError(f"Read failed: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None: raise IOError(f"Read failed: {mask_path}")

        if len(mask.shape) > 2: mask = mask[:, :, 0]

        return img, mask, img_path.stem

    def __getitem__(self, index: int, _retries: int = 0) -> Dict:
        try:
            img, mask, img_id = self._load_from_disk(index)
        except Exception as e:
            if _retries >= 10:
                raise RuntimeError(f"Failed to load any valid sample after {_retries} retries. Last error: {e}")
            print(f"Error loading {index}: {e}, retrying...")
            new_idx = random.randint(0, len(self.image_paths) - 1)
            return self.__getitem__(new_idx, _retries + 1)

        if self.transform:
            # transform internally calls get_train_transform with Resize
            img, mask = self.transform(img, mask, self.img_size)
        else:
            # Fallback logic for when no transform is provided
            target_size = self.img_size[0]
            if img.shape[0] != target_size:
                img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)

            img = torch.from_numpy(img).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).long()

        return {
            'img_id': img_id,
            'img': img,
            'gt_semantic_seg': mask
        }