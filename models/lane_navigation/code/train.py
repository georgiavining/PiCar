import tensorflow as tf
import numpy as np
from tqdm import tqdm


def train_one_epoch(model, dataset, optimiser, steps_per_epoch):
    losses = []

    for imgs, angles in tqdm(dataset, total=steps_per_epoch, desc='  train', leave=False):
        with tf.GradientTape() as tape:
            angle_pred = model(imgs, training=True)
            loss       = tf.reduce_mean(tf.square(angle_pred - angles))

        gradients = tape.gradient(loss, model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        optimiser.apply_gradients(zip(gradients, model.trainable_variables))
        losses.append(loss.numpy())

        if len(losses) >= steps_per_epoch:
            break

    return np.mean(losses)


def evaluate(model, dataset, validation_steps):
    losses = []

    for imgs, angles in tqdm(dataset, total=validation_steps, desc='  valid', leave=False):
        angle_pred = model(imgs, training=False)
        loss       = tf.reduce_mean(tf.square(angle_pred - angles))
        losses.append(loss.numpy())

        if len(losses) >= validation_steps:
            break

    return np.mean(losses)