import random
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import pandas as pd


def scan_valid_images(df, img_dir):
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


def my_imread(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def zoom(image, scale=1.3):
    h, w = image.shape[:2]
    new_h, new_w = int(h / scale), int(w / scale)
    top  = (h - new_h) // 2
    left = (w - new_w) // 2
    cropped = image[top:top+new_h, left:left+new_w]
    return cv2.resize(cropped, (w, h))


def pan(image, x_range=(-0.1, 0.1), y_range=(-0.1, 0.1)):
    h, w = image.shape[:2]
    x_shift = int(w * np.random.uniform(*x_range))
    y_shift = int(h * np.random.uniform(*y_range))
    M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    return cv2.warpAffine(image, M, (w, h))


def adjust_brightness(image, low=0.5, high=1.5):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 2] *= np.random.uniform(low, high)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def flip(image, angle):
    return cv2.flip(image, 1), 1.0 - angle


def blur(image):
    kernel_size = random.randint(1, 5)
    return cv2.blur(image, (kernel_size, kernel_size))


def random_augment(image, angle):
    if np.random.rand() < 0.5:
        image = pan(image)
    if np.random.rand() < 0.5:
        image = zoom(image)
    if np.random.rand() < 0.5:
        image = blur(image)
    if np.random.rand() < 0.5:
        image = adjust_brightness(image)
    if np.random.rand() < 0.5:
        image, angle = flip(image, angle)
    return image, angle


def preprocess(image, resize):
    h = image.shape[0]
    image = image[int(h/2):, :, :]
    image = cv2.resize(image, resize)
    return image


def image_data_generator(df, img_dir, batch_size, is_training, resize):
    img_dir     = Path(img_dir)
    image_paths = [str(img_dir / f"{int(iid)}.png") for iid in df['image_id']]
    angles      = df['angle'].values.astype(np.float32)

    while True:
        batch_images = []
        batch_angles = []

        for _ in range(batch_size):
            idx   = random.randint(0, len(image_paths) - 1)
            image = my_imread(image_paths[idx])
            angle = angles[idx]

            if is_training:
                image, angle = random_augment(image, angle)

            image = preprocess(image, resize)
            batch_images.append(image)
            batch_angles.append(angle)

        yield np.asarray(batch_images), np.asarray(batch_angles)