import random
import pandas as pd
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

DEFAULT_CROP_TOP    = 0.40
DEFAULT_CROP_BOTTOM = 0.15

SOURCE_CROPS = {
    "original"        : (0.40, 0.15),
    "oval_inner"      : (0.40, 0.18),  
    "oval_outer"      : (0.40, 0.18),
    "figure8"         : (0.40, 0.10),
    "figure8_object"  : (0.40, 0.10),
    "tjunc_left"      : (0.40, 0.18),
    "tjunc_right"     : (0.40, 0.18),
}


def scan_valid_images(df, img_dir):
    img_dir   = Path(img_dir)
    valid_ids = []
    for _, row in df.iterrows():
        image_id = str(row['image_id'])
        # original data: integer -> training_images/id.png
        # new data: full absolute path stored directly
        img_path = Path(image_id) if not image_id.isdigit() \
                   else img_dir / f"{image_id}.png"
        try:
            with Image.open(img_path) as img:
                img.load()
            valid_ids.append(row['image_id'])
        except Exception:
            print(f"Skipping corrupted image: {img_path}")
    before = len(df)
    df = df[df['image_id'].isin(set(valid_ids))].reset_index(drop=True)
    print(f"Image scan complete: {len(df)}/{before} valid")
    return df


def get_transforms(img_h, img_w, crop_top=0.40, crop_bottom=0.15):
    """
    Returns the deterministic (non-augmentation) transform pipeline.
    Augmentation is handled in CarDataset.__getitem__ so that the angle
    label can be flipped alongside the image.
    """
    crop_top_frac    = crop_top
    crop_bottom_frac = crop_bottom

    return transforms.Compose([
        transforms.Lambda(lambda img: TF.crop(
            img,
            top=int(img.height * crop_top_frac),
            left=0,
            height=int(img.height * (1 - crop_top_frac - crop_bottom_frac)),
            width=img.width,
        )),
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])


_img_augment = transforms.Compose([
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.0)), 
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
])


class CarDataset(Dataset):
    def __init__(self, df, img_dir, img_h=120, img_w=160, augment=False):
        self.df       = df.reset_index(drop=True)
        self.img_dir  = Path(img_dir)
        self.img_h    = img_h
        self.img_w    = img_w
        self.augment  = augment
        self._transform_cache = {}
        print(f"Dataset ready: {len(self.df)} images  augment={augment}")

    def _get_transform(self, crop_top, crop_bottom):
        key = (crop_top, crop_bottom)
        if key not in self._transform_cache:
            self._transform_cache[key] = get_transforms(
                self.img_h, self.img_w, crop_top, crop_bottom
            )
        return self._transform_cache[key]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        image_id = str(row['image_id'])

        if image_id.isdigit():
            img_path = self.img_dir / f"{image_id}.png"
        else:
            img_path = Path(image_id)

        img = Image.open(img_path).convert('RGB')

        source             = str(row.get('source', 'original'))
        crop_top, crop_bot = SOURCE_CROPS.get(
            source, (DEFAULT_CROP_TOP, DEFAULT_CROP_BOTTOM)
        )
        transform = self._get_transform(crop_top, crop_bot)

        angle = float(row['angle'])

        if self.augment and random.random() < 0.5:
            img   = TF.hflip(img)
            angle = 1.0 - angle

        if self.augment:
            img = _img_augment(img)

        img = transform(img)
        return img, torch.tensor(angle, dtype=torch.float32)