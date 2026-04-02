import tensorflow as tf
import numpy as np
from tqdm import tqdm


@tf.function
def train_step(model, imgs, angles, optimiser):
    angles = tf.cast(angles, tf.float32)
    angles = tf.expand_dims(angles, axis=-1)  

    with tf.GradientTape() as tape:
        preds = model(imgs, training=True)
        preds = tf.cast(preds, tf.float32)

        loss = tf.reduce_mean(tf.square(preds - angles))

    gradients = tape.gradient(loss, model.trainable_variables)
    gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    optimiser.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


@tf.function
def eval_step(model, imgs, angles):
    angles = tf.cast(angles, tf.float32)
    angles = tf.expand_dims(angles, axis=-1)  

    preds = model(imgs, training=False)
    preds = tf.cast(preds, tf.float32)

    return tf.reduce_mean(tf.square(preds - angles))


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