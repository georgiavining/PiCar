import sys 
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'lane_navigation', 'code'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'object_detection', 'code'))

from lane_model   import PiCarNetEffB2
from object_model import ObjectDetectionModel

class Driver:
    def __init__(self):
        self.lane_model = LaneModel()
        self.object_model = ObjectDetectionModel()

    def predict(self, image):
        detection = self.object_model.predict(image)

        if detection == 'pedestrian':
            return 90, 0  
        
        angle, speed = self.lane_model.predict(image)
        return angle, speed