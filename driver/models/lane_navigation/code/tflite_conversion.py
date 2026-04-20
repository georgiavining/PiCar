import tensorflow as tf
import numpy as np
import cv2
import os
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TFLITE_XNNPACK_FORCE_DISABLE'] = '1'
from pathlib import Path

CODE_DIR    = Path(__file__).resolve().parent
WEIGHTS_DIR = CODE_DIR.parent / 'outputs' / 'weights'
REPO_DIR    = Path(__file__).resolve().parents[4]
TRAIN_DIR   = REPO_DIR / 'data' / 'training_images'

def build_model(input_shape=(224, 224, 3)):
    base_model = tf.keras.applications.MobileNetV2(
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
model.load_weights(str(WEIGHTS_DIR / 'mv2_run1_best_model.h5'))

# verify before converting
img = cv2.imread(str(TRAIN_DIR / '0.png'))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224)).astype(np.float32)
pred = model(np.expand_dims(img, 0), training=False).numpy()
print(f"Pre-conversion prediction: angle={pred[0][0]*80+50:.1f} speed={round(pred[0][1])*35}")

# representative dataset
def representative_dataset():
    image_paths = sorted(Path(TRAIN_DIR).glob('*.png'))[:500]
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224)).astype(np.float32)
        img = img / 127.5 - 1.0  # MobileNetV2 preprocessing
        yield [np.expand_dims(img, 0)]

# convert
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model = converter.convert()

output_path = str(WEIGHTS_DIR / 'mv2_lane_int8_v2.tflite')
with open(output_path, 'wb') as f:
    f.write(tflite_model)
print(f"Saved to {output_path}")


interp = tf.lite.Interpreter(
    model_path=output_path,
    experimental_delegates=None,
    num_threads=1
)

interp.allocate_tensors()
inp = interp.get_input_details()
out = interp.get_output_details()
print(f"TFLite input  dtype: {inp[0]['dtype']}")
print(f"TFLite output dtype: {out[0]['dtype']}")
print(f"TFLite output quantization: {out[0]['quantization']}")

# test with same image
test_img = cv2.imread(str(TRAIN_DIR / '0.png'))
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
test_img = cv2.resize(test_img, (224, 224)).astype(np.uint8)
test_img = np.expand_dims(test_img, 0)

interp.set_tensor(inp[0]['index'], test_img)
interp.invoke()
output = interp.get_tensor(out[0]['index'])
scale, zp = out[0]['quantization']
angle_norm = (float(output[0][0]) - zp) * scale
speed_norm = (float(output[0][1]) - zp) * scale
print(f"TFLite raw output: {output[0]}")
print(f"TFLite prediction: angle={angle_norm*80+50:.1f} speed={round(speed_norm)*35}")