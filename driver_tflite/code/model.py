import os
import numpy as np
import cv2
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import detect
from pycoral.adapters import common


class LaneModel:
    saved_model  = 'lane_int8_edgetpu.tflite'
    resize_shape = (224, 224)

    def __init__(self):
        code_dir     = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(code_dir, self.saved_model)
        self.interpreter = make_interpreter(weights_path)
        self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print(f"Lane model loaded on Edge TPU from {weights_path}")

    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h     = image.shape[0]
        image = image[h//2:, :, :]
        image = cv2.resize(image, self.resize_shape)
        return np.expand_dims(image.astype(np.uint8), axis=0)

    def predict(self, image):
        img = self.preprocess(image)
        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()
        output               = self.interpreter.get_tensor(self.output_details[0]['index'])
        scale, zero_point    = self.output_details[0]['quantization']
        angle_norm           = (float(output[0][0]) - zero_point) * scale
        angle                = angle_norm * 80 + 50
        speed                = 35
        return angle, speed


class ObjectDetectionModel:
    saved_model    = 'best_full_integer_quant_edgetpu.tflite'
    classes        = ['left_turn_sign', 'right_turn_sign', 'pedestrian', 'obstacle']
    conf_threshold = 0.4

    def __init__(self):
        code_dir         = os.path.dirname(os.path.abspath(__file__))
        weights_path     = os.path.join(code_dir, self.saved_model)
        self.interpreter = make_interpreter(weights_path)
        self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print(f"Object detection model loaded on Edge TPU from {weights_path}")

    def predict(self, image):
        input_shape = self.input_details[0]['shape']
        h, w        = input_shape[1], input_shape[2]
        img         = cv2.resize(image, (w, h))
        img         = np.expand_dims(img.astype(np.uint8), axis=0)

        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()

        objs       = detect.get_objects(self.interpreter, self.conf_threshold)
        ih, iw     = image.shape[:2]
        detections = []
        for obj in objs:
            x1 = obj.bbox.xmin * iw / w
            y1 = obj.bbox.ymin * ih / h
            x2 = obj.bbox.xmax * iw / w
            y2 = obj.bbox.ymax * ih / h
            detections.append({
                'class':      self.classes[obj.id],
                'confidence': obj.score,
                'bbox':       [x1, y1, x2, y2]
            })
        return detections


class Model:
    def __init__(self):
        self.lane_model   = LaneModel()
        self.object_model = ObjectDetectionModel()

    def is_close(self, bbox, image_shape, threshold=0.05):
        x1, y1, x2, y2 = bbox
        img_h, img_w    = image_shape[:2]
        img_area        = img_h * img_w
        box_area        = (x2 - x1) * (y2 - y1)
        return (box_area / img_area) > threshold

    def is_in_road(self, bbox, image_shape, road_margin=0.6):
        x1, y1, x2, y2 = bbox
        img_w           = image_shape[1]
        box_cx          = (x1 + x2) / 2
        img_cx          = img_w / 2
        left_bound      = img_cx * (1 - road_margin / 2)
        right_bound     = img_cx * (1 + road_margin / 2)
        return left_bound < box_cx < right_bound

    def predict(self, image):
        detections = self.object_model.predict(image)

        for d in detections:
            if not self.is_close(d['bbox'], image.shape):
                continue
            in_road = self.is_in_road(d['bbox'], image.shape)
            if d['class'] in ('pedestrian', 'obstacle') and in_road:
                return 90, 0

        detected_close = [d['class'] for d in detections
                          if self.is_close(d['bbox'], image.shape)]
        if 'left_turn_sign' in detected_close or 'right_turn_sign' in detected_close:
            angle, _ = self.lane_model.predict(image)
            return angle, 20

        return self.lane_model.predict(image)