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

def flip(img, angle):
    img   = tf.image.flip_left_right(img)
    angle = 1.0 - angle
    return img, angle


def zoom(img, factor=0.3):
    h     = tf.shape(img)[0]
    w     = tf.shape(img)[1]
    scale = tf.random.uniform((), 1.0, 1.0 + factor)
    new_h = tf.cast(tf.cast(h, tf.float32) / scale, tf.int32)
    new_w = tf.cast(tf.cast(w, tf.float32) / scale, tf.int32)
    top   = (h - new_h) // 2
    left  = (w - new_w) // 2
    img   = img[top:top+new_h, left:left+new_w, :]
    img   = tf.image.resize(img, (h, w))
    return img


def pan(img, height_factor=0.1, width_factor=0.1):
    h         = tf.shape(img)[0]
    w         = tf.shape(img)[1]
    x_shift   = tf.cast(tf.cast(w, tf.float32) * tf.random.uniform((), -width_factor, width_factor), tf.int32)
    y_shift   = tf.cast(tf.cast(h, tf.float32) * tf.random.uniform((), -height_factor, height_factor), tf.int32)
    pad_top    = tf.maximum(y_shift, 0)
    pad_bottom = tf.maximum(-y_shift, 0)
    pad_left   = tf.maximum(x_shift, 0)
    pad_right  = tf.maximum(-x_shift, 0)
    img = tf.pad(img, [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
    img = tf.image.crop_to_bounding_box(img, pad_bottom, pad_right, h, w)
    return img


def adjust_brightness(img, max_delta=0.2):
    img = tf.image.random_brightness(img, max_delta=max_delta)
    img = tf.clip_by_value(img, 0, 255)
    return img

def blur(img, kernel_size=3):
    kernel = tf.ones((kernel_size, kernel_size, 3, 1), dtype=tf.float32)
    kernel = kernel / tf.cast(kernel_size * kernel_size, tf.float32)
    img = tf.expand_dims(img, 0)
    img = tf.nn.depthwise_conv2d(img, kernel, strides=[1,1,1,1], padding='SAME')
    img = tf.squeeze(img, 0)
    return img


def augment(img, angle):
    if tf.random.uniform(()) < 0.5:
        img, angle = flip(img, angle)
    if tf.random.uniform(()) < 0.5:
        img = adjust_brightness(img)
    return img, angle


def load_and_process(path, angle, resize, preprocess_fn=None):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    h = tf.shape(img)[0]
    img = img[h//2:, :, :]
    img = tf.image.resize(img, resize)
    img = tf.cast(img, tf.float32)
    if preprocess_fn is not None:
        img = preprocess_fn(img)
    else:
        img = img / 255.0
    return img, angle

def make_tf_dataset(df, img_dir, batch_size, is_training, resize, preprocess_fn=None):
    img_dir = Path(img_dir)

    paths = [str(img_dir / f"{int(i)}.png") for i in df['image_id']]
    angles = df['angle'].values.astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, angles))

    if is_training:
        ds = ds.shuffle(len(df))

    ds = ds.map(
        lambda p, a: load_and_process(p, a, resize, preprocess_fn),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.cache()

    if is_training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds