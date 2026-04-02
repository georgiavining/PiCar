import tensorflow as tf
import numpy as np
from tqdm import tqdm


def train_one_epoch(model, dataset, optimiser):
    losses = []
    for imgs, angles in tqdm(dataset, desc='  train', leave=False):
        angles = tf.cast(angles, tf.float32)
        with tf.GradientTape() as tape:
            angle_pred = tf.cast(model(imgs, training=True), tf.float32)
            loss       = tf.reduce_mean(tf.square(angle_pred - angles))
        gradients = tape.gradient(loss, model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        optimiser.apply_gradients(zip(gradients, model.trainable_variables))
        losses.append(float(loss.numpy()))
    return np.mean(losses)


def evaluate(model, dataset):
    losses = []
    for imgs, angles in tqdm(dataset, desc='  valid', leave=False):
        angles     = tf.cast(angles, tf.float32)
        angle_pred = tf.cast(model(imgs, training=False), tf.float32)
        loss       = tf.reduce_mean(tf.square(angle_pred - angles))
        losses.append(float(loss.numpy()))
    return np.mean(losses)


