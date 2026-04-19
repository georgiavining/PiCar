import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
import os

#--Paths----------------------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parent
LANE_NAV_DIR = CODE_DIR.parent
OUTPUTS_DIR = LANE_NAV_DIR / "outputs"
WEIGHTS_DIR = OUTPUTS_DIR / "weights"

REPO_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_DIR / "data"
TRAIN_CSV = DATA_DIR / "train.csv"
TRAIN_DIR = DATA_DIR / "training_images"
CACHE_DIR = os.path.join(DATA_DIR, "valid_image_ids.csv")

#--Config----------------------------------------------------------------------------
saved_model = 'mv3_angle_and_speed_best_model.h5'
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, saved_model)

#--Main------------------------------------------------------------------------------
def build_model(input_shape=(224, 224, 3)):
    base_model = tf.keras.applications.MobileNetV3Small(
        include_top=False, weights=None,
        input_shape=input_shape, pooling='avg'
    )
    base_model.trainable = True
    inputs  = tf.keras.layers.Input(shape=input_shape)
    x       = base_model(inputs, training=False)
    x       = tf.keras.layers.Dropout(0.3)(x)
    x       = tf.keras.layers.Dense(128, activation='relu')(x)
    x       = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(2, activation='sigmoid')(x)
    return tf.keras.Model(inputs, outputs)

model = build_model()
model.load_weights(WEIGHTS_PATH)

import cv2
img = cv2.imread(os.path.join(TRAIN_DIR, '0.png'))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224)).astype(np.float32)
img = np.expand_dims(img, axis=0)

pred = model(img, training=False).numpy()
print(f"Raw prediction: {pred}")
print(f"Angle: {pred[0][0] * 80 + 50:.1f}")
print(f"Speed: {round(pred[0][1]) * 35}")