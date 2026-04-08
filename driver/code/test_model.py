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

RUN_NO = 2

# ── Init driver ───────────────────────────────────────────────────────────────
from driver import Driver
driver = Driver()

# ── Debug object detection ────────────────────────────────────────────────────
print("\n── Object Detection Debug ──────────────────────────────────────────")
image_paths = sorted(Path(TEST_IMAGES_DIR).glob('*.png'))
for img_path in image_paths[:10]:
    img        = cv2.imread(str(img_path))
    detections = driver.object_model.predict(img)
    if detections:
        print(f"{img_path.name}: {detections}")
    else:
        print(f"{img_path.name}: no detections")

# ── Visual test ───────────────────────────────────────────────────────────────
print("\n── Visual Test ─────────────────────────────────────────────────────")
image_paths = sorted(Path(TEST_IMAGES_DIR).glob('*.png'))[:20]

fig, axes = plt.subplots(4, 5, figsize=(25, 20))

for ax, img_path in zip(axes.flat, image_paths):
    img = cv2.imread(str(img_path))
    angle, speed = driver.predict(img)

    h, w      = img.shape[:2]
    cx        = w // 2
    angle_rad = math.radians(angle)
    if abs(math.tan(angle_rad)) > 0.01:
        x2 = int(cx - (h // 4) / math.tan(angle_rad))
    else:
        x2 = cx
    y2 = h // 2

    color = (0, 0, 255) if speed == 0 else (0, 255, 0)  
    cv2.line(img, (cx, h), (x2, y2), color, 2)
    cv2.putText(img, f'a={angle:.1f} s={speed}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    detections = driver.object_model.predict(img)
    for d in detections:
        x1, y1, x2b, y2b = [int(v) for v in d['bbox']]
        cv2.rectangle(img, (x1, y1), (x2b, y2b), (255, 0, 0), 2)
        cv2.putText(img, f"{d['class']} {d['confidence']:.2f}",
                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(f'angle={angle:.1f} speed={speed}')
    ax.axis('off')

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, f'driver_test_run{RUN_NO}.png')
plt.savefig(output_path)
print(f"\nSaved to {output_path}")