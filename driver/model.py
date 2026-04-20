import os
import sys
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

class LaneModel:
    saved_model  = 'mv2_run1_best_model.h5'
    resize_shape = (224, 224)

    def __init__(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        WEIGHTS_PATH = os.path.join(
            BASE_DIR,
            "models/lane_navigation/outputs/weights/" + self.saved_model
        )
        self.model   = self._build_model()
        self.model.load_weights(WEIGHTS_PATH)
        self.model.trainable = False
        print(f"Lane model loaded from {WEIGHTS_PATH}")

    def _build_model(self, input_shape=(224, 224, 3)):
        base_model = tf.keras.applications.MobileNetV2Small(
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
        image = image / 127.5 - 1.0                    
        image = tf.expand_dims(image, axis=0)
        return image

    def predict(self, image):
        img    = self.preprocess(image)
        output = self.model(img, training=False).numpy()[0]
        angle  = float(output[0]) * 80 + 50
        speed  = round(float(output[1])) * 35
        return angle, speed


class ObjectDetectionModel:
    saved_model = 'run3_best.pt'
    classes = ['left_turn_sign', 'right_turn_sign', 'pedestrian', 'obstacle']
    conf_threshold = 0.4

    def __init__(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        WEIGHTS_PATH = os.path.join(
            BASE_DIR,
            "../../models/object_detection_weights/" + self.saved_model
        )
        self.model = YOLO(WEIGHTS_PATH)
        print(f"Object detection model loaded from {WEIGHTS_PATH}")

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

class CarState:
    LANE_FOLLOWING = 'lane_following'
    STOPPING       = 'stopping'
    TURNING_LEFT   = 'turning_left'
    TURNING_RIGHT  = 'turning_right'


class Model:
    def __init__(self):
        self.state        = CarState.LANE_FOLLOWING
        self.lane_model   = LaneModel()
        self.object_model = ObjectDetectionModel()

    def is_close(self, bbox, image_shape, threshold=0.025, vertical_threshold=0.6):
        """
        Object is close if either:
        - bbox area is large enough (close object)
        - OR bbox bottom is in lower half of image (near object)
        """
        x1, y1, x2, y2 = bbox
        img_h, img_w    = image_shape[:2]
        
        img_area  = img_h * img_w
        box_area  = (x2 - x1) * (y2 - y1)
        area_frac = box_area / img_area

        y2_frac = y2 / img_h  
        
        return (area_frac > threshold) or (y2_frac > vertical_threshold)

    def is_in_road(self, bbox, image_shape, road_margin=0.3):
        x1, y1, x2, y2 = bbox
        img_w           = image_shape[1]
        box_cx          = (x1 + x2) / 2
        img_cx          = img_w / 2
        left_bound      = img_cx * (1 - road_margin / 2)
        right_bound     = img_cx * (1 + road_margin / 2)
        return left_bound < box_cx < right_bound

    def _update_state(self, detections, image_shape):
        close = [d for d in detections if self.is_close(d['bbox'], image_shape)]

        for d in close:
            in_road  = self.is_in_road(d['bbox'], image_shape)
            x1,y1,x2,y2 = d['bbox']
            ih, iw   = image_shape[:2]
            area_frac = (x2-x1)*(y2-y1) / (ih*iw)
            y2_frac  = y2 / ih
            print(f"  {d['class']} conf={d['confidence']:.2f} area={area_frac:.3f} y2={y2_frac:.2f} in_road={in_road}")

        if any(d['class'] in ('pedestrian', 'obstacle')
            and self.is_in_road(d['bbox'], image_shape)
            for d in close):
            self.state = CarState.STOPPING
            print(f"  → STOPPING")
        elif any(d['class'] == 'left_turn_sign' for d in close):
            self.state = CarState.TURNING_LEFT
            print(f"  → TURNING_LEFT")
        elif any(d['class'] == 'right_turn_sign' for d in close):
            self.state = CarState.TURNING_RIGHT
            print(f"  → TURNING_RIGHT")
        else:
            self.state = CarState.LANE_FOLLOWING

    def predict(self, image):
        detections = self.object_model.predict(image)
        self._update_state(detections, image.shape)

        if self.state == CarState.STOPPING:
            return 90, 0
        elif self.state == CarState.TURNING_LEFT:
            angle, _ = self.lane_model.predict(image)
            return angle, 20
        elif self.state == CarState.TURNING_RIGHT:
            angle, _ = self.lane_model.predict(image)
            return angle, 20
        else:
            return self.lane_model.predict(image)