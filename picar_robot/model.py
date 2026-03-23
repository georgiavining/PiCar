import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import os
from src.model import PiCarNet

class Model:
    def __init__(self):
        model_dir = os.path.dirname(os.path.abspath(__file__))
        self.model = PiCarNet()
        self.model.load_state_dict(
            torch.load(
                os.path.join(model_dir,'..', 'outputs', 'models', 'mobilenet_v3_small_baseline_best_model.pth'),
                map_location='cpu'
            )
        )
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((120, 160)),  
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]
            ),
        ])

    def predict(self, image):
        img = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            pred = self.model(img)
        angle_norm = float(pred[0, 0])
        speed_norm = float(pred[0, 1])

        angle = angle_norm * 80 + 50    
        speed = round(speed_norm) * 35  
        return angle, speed