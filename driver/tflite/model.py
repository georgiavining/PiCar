import os
import numpy as np
import cv2
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import detect
from pycoral.adapters import common


class LaneModel:
    saved_model  = 'mv2_lane_int8_edgetpu.tflite'
    resize_shape = (224, 224)

    def __init__(self):
        code_dir     = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(code_dir, self.saved_model)
        
        try:
            self.interpreter = make_interpreter(weights_path)
            print('Lane model: Using TPU')
        except Exception as e:
            print(f'Lane model: Fallback to CPU — {e}')
            import tflite_runtime.interpreter as tflite
            self.interpreter = tflite.Interpreter(weights_path)
        
        self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print(f"Lane model loaded from {weights_path}")

    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.resize_shape)
        # MobileNetV2 expects uint8 input [0, 255] when quantized
        image = image.astype(np.uint8)
        return np.expand_dims(image, axis=0)

    def predict(self, image):
        img = self.preprocess(image)
        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()
        output            = self.interpreter.get_tensor(self.output_details[0]['index'])
        scale, zero_point = self.output_details[0]['quantization']
        angle_norm        = (float(output[0][0]) - zero_point) * scale
        speed_norm        = (float(output[0][1]) - zero_point) * scale
        angle             = angle_norm * 80 + 50
        speed             = round(speed_norm) * 35
        return angle, speed


class ObjectDetectionModel:
    saved_model    = 'best_full_integer_quant_edgetpu.tflite'
    classes        = ['left_turn_sign', 'right_turn_sign', 'pedestrian', 'obstacle']
    conf_threshold = 0.4

    def __init__(self):
        code_dir         = os.path.dirname(os.path.abspath(__file__))
        weights_path     = os.path.join(code_dir, self.saved_model)
        
        try:
            self.interpreter = make_interpreter(weights_path)
            print('Lane model: Using TPU')
        except Exception as e:
            print(f'Lane model: Fallback to CPU — {e}')
            import tflite_runtime.interpreter as tflite
            self.interpreter = tflite.Interpreter(weights_path)
        
        self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print(f"Object detection model loaded from {weights_path}")

    def predict(self, image):
        input_shape = self.input_details[0]['shape']
        h, w        = input_shape[1], input_shape[2]
        img         = cv2.resize(image, (w, h))
        img         = (img.astype(np.int16) - 128).astype(np.int8)
        img         = np.expand_dims(img, axis=0)

        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()

        # dequantize output
        output            = self.interpreter.get_tensor(self.output_details[0]['index'])
        scale, zero_point = self.output_details[0]['quantization']
        output            = (output.astype(np.float32) - zero_point) * scale
        # output shape: (1, 8, 2100) — transpose to (1, 2100, 8)
        output = output[0].T  # shape (2100, 8)

        ih, iw     = image.shape[:2]
        detections = []

        for row in output:
            class_scores = row[4:]                    # 4 class scores
            cls_id       = int(np.argmax(class_scores))
            confidence   = float(class_scores[cls_id])

            if confidence < self.conf_threshold:
                continue

            # bbox is cx, cy, bw, bh normalised to input size
            cx, cy, bw, bh = row[:4]
            x1 = (cx - bw / 2) * iw / w
            y1 = (cy - bh / 2) * ih / h
            x2 = (cx + bw / 2) * iw / w
            y2 = (cy + bh / 2) * ih / h

            detections.append({
                'class':      self.classes[cls_id],
                'confidence': confidence,
                'bbox':       [x1, y1, x2, y2]
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

    def is_close(self, bbox, image_shape, threshold=0.025):
        x1, y1, x2, y2 = bbox
        img_h, img_w    = image_shape[:2]
        box_area        = (x2 - x1) * (y2 - y1)
        return (box_area / (img_h * img_w)) > threshold

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

        if any(d['class'] in ('pedestrian', 'obstacle')
               and self.is_in_road(d['bbox'], image_shape)
               for d in close):
            self.state = CarState.STOPPING
        elif any(d['class'] == 'left_turn_sign' for d in close):
            self.state = CarState.TURNING_LEFT
        elif any(d['class'] == 'right_turn_sign' for d in close):
            self.state = CarState.TURNING_RIGHT
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