import os
import sys
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

class LaneModel:
    saved_model  = 'mv3_angle_and_speed_best_model.h5'
    resize_shape = (224, 224)

    def __init__(self):
        code_dir     = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(code_dir, self.saved_model)
        self.model   = self._build_model()
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
        outputs = tf.keras.layers.Dense(2, activation='sigmoid')(x)
        return tf.keras.Model(inputs, outputs)

    def preprocess(self, image):
        image = tf.convert_to_tensor(image, dtype=tf.uint8)
        image = image[:, :, ::-1]
        image = tf.image.resize(image, self.resize_shape)
        image = tf.cast(image, tf.float32)
        image = tf.expand_dims(image, axis=0)
        return image

    def predict(self, image):
        img    = self.preprocess(image)
        output = self.model(img, training=False).numpy()[0]
        angle  = float(output[0]) * 80 + 50
        speed  = round(float(output[1])) * 35  
        return angle, speed

class ObjectDetectionModel:
    saved_model = 'best_model.pt'
    classes = ['left_turn_sign', 'right_turn_sign', 'pedestrian', 'obstacle']
    conf_threshold = 0.4

    def __init__(self):
        code_dir    = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(code_dir, self.saved_model)
        self.model = YOLO(weights_path)
        print(f"Object detection model loaded from {weights_path}")

    def predict(self, image):
        """
        Run object detection on a BGR camera frame.
        Returns list of detected class names above confidence threshold.
        """
        results = self.model(image, conf=self.conf_threshold, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                name   = self.classes[cls_id]
                detections.append({
                    'class': name,
                    'confidence': conf,
                    'bbox': box.xyxy[0].tolist()  
                })
        return detections


class Model:
    def __init__(self):
        self.lane_model   = LaneModel()
        self.object_model = ObjectDetectionModel()

    def is_close(self, bbox, image_shape, threshold=0.05):
        """
        Check if detected object is close enough to act on.
        Uses bounding box area as proxy for distance.
        threshold: fraction of image area — tune this based on testing
        """
        x1, y1, x2, y2 = bbox
        img_h, img_w    = image_shape[:2]
        img_area        = img_h * img_w
        box_area        = (x2 - x1) * (y2 - y1)
        return (box_area / img_area) > threshold
    
    def is_in_road(self, bbox, image_shape, road_margin=0.6):
        """
        Check if object is in the road (centre) vs side.
        road_margin: fraction of image width considered 'in road'
        """
        x1, y1, x2, y2 = bbox
        img_w   = image_shape[1]
        box_cx  = (x1 + x2) / 2  
        img_cx  = img_w / 2       

        left_bound  = img_cx * (1 - road_margin / 2)
        right_bound = img_cx * (1 + road_margin / 2)

        return left_bound < box_cx < right_bound

    def predict(self, image):
        detections = self.object_model.predict(image)

        for d in detections:
            if not self.is_close(d['bbox'], image.shape):
                continue                                    

            in_road = self.is_in_road(d['bbox'], image.shape)

            if d['class'] == 'pedestrian' and in_road:
                return 90, 0                                

            if d['class'] == 'obstacle' and in_road:
                return 90, 0                                

        detected_close = [d['class'] for d in detections
                        if self.is_close(d['bbox'], image.shape)]
        if 'left_turn_sign' in detected_close or 'right_turn_sign' in detected_close:
            angle, _ = self.lane_model.predict(image)
            return angle, 20

        return self.lane_model.predict(image)