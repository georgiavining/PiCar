import tensorflow as tf
import numpy as np
from tqdm import tqdm


@tf.function
def train_step(model, imgs, angles, optimiser):
    with tf.GradientTape() as tape:
        angle_pred = tf.cast(model(imgs, training=True), tf.float32)
        angles     = tf.cast(angles, tf.float32)
        loss       = tf.reduce_mean(tf.square(angle_pred - angles))
    gradients = tape.gradient(loss, model.trainable_variables)
    gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    optimiser.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


@tf.function
def eval_step(model, imgs, angles):
    angle_pred = tf.cast(model(imgs, training=False), tf.float32)
    angles     = tf.cast(angles, tf.float32)
    return tf.reduce_mean(tf.square(angle_pred - angles))


def train_one_epoch(model, dataset, optimiser):
    losses = []
    for imgs, angles in tqdm(dataset, desc='  train', leave=False):
        loss = train_step(model, imgs, angles, optimiser)
        losses.append(loss.numpy())
    return np.mean(losses)


def evaluate(model, dataset):
    losses = []
    for imgs, angles in tqdm(dataset, desc='  valid', leave=False):
        loss = eval_step(model, imgs, angles)
        losses.append(loss.numpy())
    return np.mean(losses)