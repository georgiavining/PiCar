import tensorflow as tf
import numpy as np
import os

def build_dataset(filenames, labels=None, img_size=224, batch_size=32,
                  shuffle=False, shuffle_buffer=1000):
    """
    Single function for all dataset types:
    - labels=None         → test dataset (no labels)
    - labels=array        → single output dataset (angle or speed)
    - labels=(arr1, arr2) → dual output dataset (angle + speed dict)
    """

    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(filenames)
        ds = ds.map(lambda f: preprocess_image(f, img_size), num_parallel_calls=2)
    elif isinstance(labels, tuple):
        ds = tf.data.Dataset.from_tensor_slices((filenames, labels[0], labels[1]))
        ds = ds.map(lambda f, a, s: (preprocess_image(f, img_size), {"angle": a, "speed": s}),
                    num_parallel_calls=2)
    else:
        ds = tf.data.Dataset.from_tensor_slices((filenames, labels))
        ds = ds.map(lambda f, l: (preprocess_image(f, img_size), l), num_parallel_calls=2)

    if shuffle:
        ds = ds.shuffle(shuffle_buffer)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def split_indices(n_total, val_split=0.1, test_split=0.1):
    idx = np.random.permutation(n_total)
    n_test = int(n_total * test_split)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val - n_test
    return idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]

def remove_invalid_images(df, image_dir):
    invalid_images = [idx for idx in df['image_id'] 
                      if not os.path.exists(os.path.join(image_dir, f"{idx}.png")) 
                      or os.path.getsize(os.path.join(image_dir, f"{idx}.png")) == 0]
    
    df = df[~df['image_id'].isin(invalid_images)]

    return df, invalid_images

def preprocess_image(filename, img_size):
    img = tf.io.read_file(filename)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [img_size, img_size])
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

