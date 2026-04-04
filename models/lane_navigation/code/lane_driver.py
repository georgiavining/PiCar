import os
import sys
import numpy as np
import tensorflow as tf



class LaneModel:
    saved_model = 'mv3_run7_best_model.h5'
    resize_shape = (224, 224)

    def __init__(self):
        code_dir    = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(code_dir, '..', 'outputs', 'weights', self.saved_model)
        self.model = self._build_model()
        self.model.load_weights(weights_path)
        self.model.trainable = False
        print(f"Lane model loaded from {weights_path}")

    def _build_model(self, input_shape=(224, 224, 3)):
        base_model = tf.keras.applications.MobileNetV3Small(
            include_top=False,
            weights=None,           
            input_shape=input_shape,
            pooling='avg'
        )
        base_model.trainable = True
        inputs  = tf.keras.layers.Input(shape=input_shape)
        x       = base_model(inputs, training=False)
        x       = tf.keras.layers.Dropout(0.3)(x)
        x       = tf.keras.layers.Dense(128, activation='relu')(x)
        x       = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        return tf.keras.Model(inputs, outputs)

    def preprocess(self, image):
        image = tf.convert_to_tensor(image, dtype=tf.uint8)  
        image = image[:, :, ::-1]
        h= tf.shape(image)[0]
        image = image[h//2:, :, :]                            
        image = tf.image.resize(image, self.resize_shape)
        image = tf.cast(image, tf.float32)
        image = tf.expand_dims(image, axis=0)
        return image

    def predict(self, image):
        img      = self.preprocess(image)
        angle_norm = float(self.model(img, training=False).numpy()[0][0])
        angle    = angle_norm * 80 + 50         
        speed    = 35                           
        return angle, speed