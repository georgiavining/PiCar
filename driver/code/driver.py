import os
import sys
import numpy as np
import tensorflow as tf

DRIVER_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR        = os.path.dirname(os.path.dirname(DRIVER_CODE_DIR))

sys.path.append(os.path.join(REPO_DIR, 'models', 'lane_navigation', 'code'))
sys.path.append(os.path.join(REPO_DIR, 'models', 'object_detection', 'code'))

from lane_driver  import LaneModel
from object_driver import ObjectDetectionModel


class Driver:
    def __init__(self):
        self.lane_model   = LaneModel()
        self.object_model = ObjectDetectionModel()

    def is_close(self, bbox, image_shape, threshold=0.15):
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
        e.g. 0.6 means middle 60% of image width
        """
        x1, y1, x2, y2 = bbox
        img_w   = image_shape[1]
        box_cx  = (x1 + x2) / 2  
        img_cx  = img_w / 2       

        left_bound  = img_cx * (1 - road_margin / 2)
        right_bound = img_cx * (1 + road_margin / 2)

        return left_bound < box_cx < right_bound

    def predict(self, image):
        detections       = self.object_model.predict(image)
        detected_classes = [d['class'] for d in detections
                            if self.is_close(d['bbox'], image.shape)]

        if 'pedestrian' in detected_classes:
            return 90, 0

        if 'obstacle' in detected_classes:
            return 90, 0

        if 'left_turn_sign' in detected_classes or 'right_turn_sign' in detected_classes:
            angle, _ = self.lane_model.predict(image)
            return angle, 20

        angle, speed = self.lane_model.predict(image)
        return angle, speed