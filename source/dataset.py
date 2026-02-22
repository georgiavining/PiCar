import tensorflow as tf
import numpy as np

def preprocess_image(filename, img_size):
    img = tf.io.read_file(filename)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [img_size, img_size])
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def build_train_parser(img_size):
    def parse_image(filename, angle, speed):
        img = preprocess_image(filename, img_size)
        return img, {"angle": angle, "speed": speed}
    return parse_image


def build_test_parser(img_size):
    def parse_image(filename):
        img = preprocess_image(filename, img_size)
        return img
    return parse_image

def create_dataset(
    filenames,
    angles=None,
    speeds=None,
    img_size=224,
    batch_size=32,
    shuffle=True,
    sample_size=None,
    val_split=0.0,
    test_split=0.0,
    seed=42,
    cache=False
):
    """
    Create TensorFlow datasets for train/validation/test.

    Returns:
        If val_split=test_split=0 → single dataset
        Otherwise → (train_ds, val_ds, test_ds)
    """

    if sample_size is not None:
        filenames = filenames[:sample_size]
        if angles is not None:
            angles = angles[:sample_size]
            speeds = speeds[:sample_size]

    #test dataset doesn't have labels, so we handle that case separately
    if angles is None or speeds is None:
        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        dataset = dataset.map(
            build_test_parser(img_size),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    if val_split > 0 or test_split > 0:
        np.random.seed(seed)
        idx = np.random.permutation(len(filenames))

        n_total = len(filenames)
        n_test = int(n_total * test_split)
        n_val = int(n_total * val_split)
        n_train = n_total - n_val - n_test

        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]

        def build_subset(indices, shuffle_flag):
            ds = tf.data.Dataset.from_tensor_slices((
                np.array(filenames)[indices],
                np.array(angles)[indices],
                np.array(speeds)[indices]
            ))

            ds = ds.map(
                build_train_parser(img_size),
                num_parallel_calls=tf.data.AUTOTUNE
            )

            if cache:
                ds = ds.cache()

            if shuffle_flag:
                shuffle_buffer = len(indices)
                ds = ds.shuffle(shuffle_buffer)

            ds = ds.batch(batch_size)

            return ds.prefetch(tf.data.AUTOTUNE)

        train_ds = build_subset(train_idx, shuffle)
        val_ds = build_subset(val_idx, False)
        test_ds = build_subset(test_idx, False)

        return train_ds, val_ds, test_ds

    # If no splits, return single dataset
    dataset = tf.data.Dataset.from_tensor_slices((filenames, angles, speeds))
    dataset = dataset.map(
        build_train_parser(img_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if cache:
        dataset = dataset.cache()

    if shuffle:
        shuffle_buffer = len(filenames)
        dataset = dataset.shuffle(shuffle_buffer)

    dataset = dataset.batch(batch_size)

    return dataset.prefetch(tf.data.AUTOTUNE)
