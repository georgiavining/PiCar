import os
import numpy as np
from ultralytics import YOLO

class ObjectDetectionModel:
    saved_model = 'best.pt'
    classes = ['left_turn_sign', 'right_turn_sign', 'pedestrian', 'obstacle']
    conf_threshold = 0.4

    def __init__(self):
        code_dir    = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(code_dir, '..', 'outputs', 'weights', self.saved_model)
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