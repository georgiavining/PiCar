import random
import numpy as np
from pathlib import Path
from PIL import Image
import pandas as pd
import tensorflow as tf

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

def load_and_process(path, angle, resize):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    h   = tf.shape(img)[0]
    img = img[h//3:, :, :]
    img = tf.image.resize(img, resize)
    img = tf.cast(img, tf.float32)
    return img, angle

def zoom(img, factor=0.3):
    img = tf.keras.layers.RandomZoom(
        height_factor=(-factor, 0.0),  
        fill_mode='nearest'
    )(img[tf.newaxis, ...])[0]
    return img

def pan(img, height_factor=0.1, width_factor=0.1):
    img = tf.keras.layers.RandomTranslation(
        height_factor=height_factor, width_factor=width_factor,
        fill_mode='nearest'
    )(img[tf.newaxis, ...])[0]
    return img

def adjust_brightness(img, max_delta=50.0):
    img = tf.image.random_brightness(img, max_delta=max_delta)
    img = tf.clip_by_value(img, 0, 255)
    return img

def flip(img, angle):
    img   = tf.image.flip_left_right(img)
    angle = 1.0 - angle
    return img, angle


def random_augment(img, angle):
    if np.random.rand() < 0.5:
        img = pan(img)
    if np.random.rand() < 0.5:
        img = adjust_brightness(img)
    if np.random.rand() < 0.5:
        img = zoom(img)
    if np.random.rand() < 0.5:
        img, angle = flip(img, angle)
    return img, angle


def make_tf_dataset(df, img_dir, batch_size, is_training, resize):
    img_dir     = Path(img_dir)
    image_paths = [str(img_dir / f"{int(iid)}.png") for iid in df['image_id']]
    angles      = df['angle'].values.astype(np.float32)

    path_ds = tf.data.Dataset.from_tensor_slices((image_paths, angles))

    if is_training:
        path_ds = path_ds.shuffle(buffer_size=len(df), reshuffle_each_iteration=True)

    ds = path_ds.map(
        lambda path, angle: load_and_process(path, angle, resize),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.cache()

    if is_training:
        ds = ds.map(random_augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds