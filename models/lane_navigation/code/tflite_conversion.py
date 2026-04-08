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
saved_model = 'mv3_run7_best_model.h5'
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
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs, outputs)

model = build_model()
model.load_weights(WEIGHTS_PATH)


def representative_dataset():
    image_paths = list(Path(TRAIN_DIR).glob('*.png'))[:200]
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h   = img.shape[0]
        img = img[h//2:, :, :]
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32)
        yield [np.expand_dims(img, axis=0)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model = converter.convert()
with open(os.path.join(WEIGHTS_DIR, 'lane_int8.tflite'), 'wb') as f:
    f.write(tflite_model)
print("Lane TFLite model saved to Drive")