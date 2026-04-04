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
        self.model = tf.keras.models.load_model(weights_path)
        self.model.trainable = False
        print(f"Lane model loaded from {weights_path}")

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