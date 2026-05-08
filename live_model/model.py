import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
import os


class PiCarNet(nn.Module):
    def __init__(self, pretrained=True, dropout_rate_first=0.3, dropout_rate_second=0.2):
        super().__init__()
        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        )
        in_features = backbone.classifier[0].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(p=dropout_rate_first),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate_second),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


class Model:
    # ── Crop params ────────────────────────────────────
    CROP_TOP    = 0.40
    CROP_BOTTOM = 0.15

    # ── Obstacle detection ───────────────────────────────────────────────────
    # ROI applied after crop+resize to 120x160
    # Centre third horizontally, lower half vertically
    ROI_TOP   = 50
    ROI_LEFT  = 55
    ROI_RIGHT = 105

    # HSV floor definition: bright, low-saturation (white/light grey tile)
    FLOOR_S_MAX = 40
    FLOOR_V_MIN = 180

    # Non-floor pixel count to trigger stop
    OBSTACLE_THRESHOLD = 800

    # Only run obstacle check when driving roughly straight
    STRAIGHT_LOW  = 0.4
    STRAIGHT_HIGH = 0.6

    DEFAULT_SPEED = 35

    def __init__(self):
        model_dir = os.path.dirname(os.path.abspath(__file__))
        self.model = PiCarNet(pretrained=False)
        self.model.load_state_dict(
            torch.load(
                os.path.join(model_dir, 'finetune_mv3_best_model.pth'),
                map_location='cpu'
            )
        )
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda img: TF.crop(
                img,
                top=int(img.height * self.CROP_TOP),
                left=0,
                height=int(img.height * (1 - self.CROP_TOP - self.CROP_BOTTOM)),
                width=img.width,
            )),
            transforms.Resize((120, 160)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225],
            ),
        ])

    def _obstacle_in_path(self, image_rgb: np.ndarray) -> bool:
        """
        Detects non-floor objects in the centre-lane ROI.
        image_rgb: raw frame (H x W x 3, uint8, RGB) before any crop/resize.
        """
        h, w = image_rgb.shape[:2]
        top    = int(h * self.CROP_TOP)
        bottom = int(h * (1 - self.CROP_BOTTOM))
        cropped = image_rgb[top:bottom, :]
        resized = cv2.resize(cropped, (160, 120))

        # Extract centre ROI
        roi = resized[self.ROI_TOP:, self.ROI_LEFT:self.ROI_RIGHT]

        # Classify floor pixels in HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        floor_mask    = (hsv[:, :, 1] <= self.FLOOR_S_MAX) & \
                        (hsv[:, :, 2] >= self.FLOOR_V_MIN)
        non_floor_pixels = int((~floor_mask).sum())

        return non_floor_pixels > self.OBSTACLE_THRESHOLD

    def predict(self, image: np.ndarray):
        img_tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            pred = self.model(img_tensor)
        angle_norm = float(pred[0, 0])
        angle = angle_norm * 80 + 50
        speed = self.DEFAULT_SPEED

        if self.STRAIGHT_LOW < angle_norm < self.STRAIGHT_HIGH:
            if self._obstacle_in_path(image):
                speed = 0

        return angle, speed