# Steering Angle & Speed Prediction
> Machine Learning in Science II 2026 — Kaggle Competition

A deep learning project to predict **steering angle** and **speed** from dashcam images using TensorFlow CNNs.

---

## Folder Structure

```
kaggle/
├── data/
│   └── train.csv               # Training labels (image_id, angle, speed)
├── notebooks/
│   └── images/                          # Training plots of MSE over epochs
│   └── mlis2026-cnn-angle-speed.ipynb   # Main experiments notebook
│   └──train.ipynb                       # Training notebook (Google Colab)
└── outputs/
    ├── models/                 # Saved model checkpoints (.keras)
    └── predictions/            # Submission CSV files
```

---

## Dataset

- **Training images:** 14,383 dashcam images (224×224 PNG)
- **Test images:** 2,000 images
- **Targets:**
  - `angle` — continuous, normalised to [0, 1]
  - `speed` — binary (0 = stopped, 1 = moving)
- **Class imbalance:** 80% speed=1, 20% speed=0
- **Evaluation metric:** Mean Squared Error (MSE), averaged across angle and speed

---

## Key Design Decisions

- **Separate models** for angle and speed — biggest performance improvement, as the tasks require different feature representations
- **Weighted binary cross-entropy** for speed — handles 80/20 class imbalance (w₀=2.4, w₁=0.6)
- **GlobalAveragePooling2D** instead of Flatten — reduces parameters and overfitting
- **BatchNormalisation** after every conv layer — stabilises training
- **ReduceLROnPlateau** — adaptive learning rate instead of manual LR search
