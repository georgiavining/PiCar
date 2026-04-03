from pathlib import Path
import os
import pandas as pd
import tensorflow as tf
from PIL import Image
import numpy as np

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

def augment(img, angle):
    if tf.random.uniform(()) < 0.5:
        img = tf.image.flip_left_right(img)
        angle = 1.0 - angle

    if tf.random.uniform(()) < 0.5:
        img = tf.image.random_brightness(img, 0.2)

    return img, angle

def load_and_process(path, angle, resize, scale=True):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    h   = tf.shape(img)[0]
    img = tf.image.resize(img, resize)
    img = tf.cast(img, tf.float32)
    if scale:
      img = img/255.
    return img, angle

def make_tf_dataset(df, img_dir, batch_size, is_training, resize):
    img_dir = Path(img_dir)

    paths = [str(img_dir / f"{int(i)}.png") for i in df['image_id']]
    angles = df['angle'].values.astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, angles))

    if is_training:
        ds = ds.shuffle(len(df))

    ds = ds.map(
        lambda p, a: load_and_process(p, a, resize),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if is_training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds