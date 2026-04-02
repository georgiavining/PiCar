import random
import numpy as np
from pathlib import Path
from PIL import Image
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

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
    img = preprocess_input(img)
    return img, angle

def zoom(img, factor=0.3):
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    scale = tf.random.uniform((), 1.0, 1.0 + factor)
    new_h = tf.cast(tf.cast(h, tf.float32) / scale, tf.int32)
    new_w = tf.cast(tf.cast(w, tf.float32) / scale, tf.int32)
    top   = (h - new_h) // 2
    left  = (w - new_w) // 2
    img   = img[top:top+new_h, left:left+new_w, :]
    img   = tf.image.resize(img, (h, w))
    return img

def pan(img, height_factor=0.1, width_factor=0.1):
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    x_shift = tf.cast(tf.cast(w, tf.float32) * tf.random.uniform((), -width_factor, width_factor), tf.int32)
    y_shift = tf.cast(tf.cast(h, tf.float32) * tf.random.uniform((), -height_factor, height_factor), tf.int32)
    
    pad_top    = tf.maximum(y_shift, 0)
    pad_bottom = tf.maximum(-y_shift, 0)
    pad_left   = tf.maximum(x_shift, 0)
    pad_right  = tf.maximum(-x_shift, 0)

    img = tf.pad(img, [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
    img = tf.image.crop_to_bounding_box(img, pad_bottom, pad_right, h, w)
    return img

def color_jitter(img):
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    return img

def flip(img, angle):
    img   = tf.image.flip_left_right(img)
    angle = 1.0 - angle
    return img, angle


def random_augment(img, angle):
    if tf.random.uniform(()) < 0.5:
        img = pan(img)

    if tf.random.uniform(()) < 0.5:
        img = zoom(img)

    if tf.random.uniform(()) < 0.5:
        img = color_jitter(img)

    if tf.random.uniform(()) < 0.5:
        img, angle = flip(img, angle)

    return img, angle


def make_tf_dataset(df, img_dir, batch_size, is_training, resize):
    img_dir = Path(img_dir)

    image_paths = [str(img_dir / f"{int(iid)}.png") for iid in df['image_id']]
    angles = df['angle'].values.astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((image_paths, angles))

    if is_training:
        ds = ds.shuffle(len(df))

    ds = ds.map(
        lambda p, a: load_and_process(p, a, resize),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if is_training:
        ds = ds.map(random_augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)