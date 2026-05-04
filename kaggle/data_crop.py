import pandas as pd
import os
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def scan_valid_images(df: pd.DataFrame, img_dir: str) -> pd.DataFrame:
    img_dir   = Path(img_dir)
    valid_ids = []
    for _, row in df.iterrows():
        img_path = img_dir / f"{int(row['image_id'])}.png"
        try:
            with Image.open(img_path) as img:
                img.load()
            valid_ids.append(row["image_id"])
        except Exception:
            print(f"Skipping corrupted image: {img_path}")
    before = len(df)
    df     = df[df["image_id"].isin(set(valid_ids))].reset_index(drop=True)
    print(f"Image scan complete: {len(df)}/{before} valid")
    return df

def get_transforms(augment, img_h, img_w, crop_top=0.3):
    base = [
        transforms.Lambda(lambda img: transforms.functional.crop(
            img,
            top=int(img.height * crop_top),
            left=0,
            height=int(img.height * (1 - crop_top)),
            width=img.width
        )),
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]
    if augment:
        aug = [
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        ]
        return transforms.Compose(aug + base)
    return transforms.Compose(base)

class CarDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = Path(img_dir)
        self.transform = transform
        self.is_test   = is_test
        print(f"Dataset ready: {len(self.df)} images")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = self.img_dir / f"{int(row['image_id'])}.png"
        img      = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        if self.is_test:
            return img, int(row['image_id'])
        label = torch.tensor(
            [float(row['angle']), float(row['speed'])],
            dtype=torch.float32
        )
        return img, label
