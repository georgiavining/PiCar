import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math

CODE_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_DIR  = os.path.dirname(os.path.dirname(CODE_DIR))
sys.path.append(CODE_DIR)

TEST_IMAGES_DIR = os.path.join(REPO_DIR, 'data', 'test_images')
OUTPUT_DIR      = os.path.join(REPO_DIR, 'driver', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

from model import Model
driver = Model()

image_paths = sorted(Path(TEST_IMAGES_DIR).glob('*.png'))

stopping_images = []

for img_path in image_paths:
    img           = cv2.imread(str(img_path))
    angle, speed  = driver.predict(img)
    detections    = driver.object_model.predict(img)

    if speed == 0:
        # draw bounding boxes and info
        for d in detections:
            x1, y1, x2, y2 = [int(v) for v in d['bbox']]
            ih, iw          = img.shape[:2]
            box_area        = (x2 - x1) * (y2 - y1)
            img_area        = ih * iw
            area_frac       = box_area / img_area
            in_road         = driver.is_in_road(d['bbox'], img.shape)

            color = (0, 0, 255) if in_road else (255, 165, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img,
                f"{d['class']} area={area_frac:.3f} road={in_road}",
                (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        cv2.putText(img, f'STOPPED a={angle:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        stopping_images.append((img_path.name, img))

print(f"Found {len(stopping_images)} stopping images")

n     = len(stopping_images)
ncols = 4
nrows = math.ceil(n / ncols)

if n > 0:
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows))
    axes = axes.flat if n > 1 else [axes]

    for ax, (name, img) in zip(axes, stopping_images):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(name, fontsize=8)
        ax.axis('off')

    for ax in list(axes)[n:]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'stopping_images.png'))
    print(f"Saved to {OUTPUT_DIR}/stopping_images.png")
else:
    print("No stopping images found — try lowering is_close threshold")