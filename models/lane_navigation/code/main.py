import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from lane_model import PiCarNet
from data import scan_valid_images, image_data_generator
from train import train_one_epoch, evaluate

# ── Config ────────────────────────────────────────────────────────────────────
SEED           = 42
IMG_H          = 260
IMG_W          = 260
BATCH_SIZE     = 32
EPOCHS         = 60
LR             = 2e-4
PATIENCE       = 12
DROPOUT_FIRST  = 0.3
DROPOUT_SECOND = 0.15
RUN_NAME       = "effb2_angle_run1"

# ── Paths ─────────────────────────────────────────────────────────────────────
CODE_DIR            = os.path.dirname(os.path.abspath(__file__))          
MODEL_DIR           = os.path.dirname(CODE_DIR)                          
REPO_DIR            = os.path.dirname(os.path.dirname(MODEL_DIR))        

DATA_PATH           = os.path.join(REPO_DIR, "kaggle_data")               
TRAIN_CSV           = os.path.join(DATA_PATH, "train.csv")
TRAIN_DIR           = os.path.join(DATA_PATH, "training_images")               
OUTPUTS_PATH        = os.path.join(MODEL_DIR, "outputs") 
WEIGHTS_DIR         = os.path.join(OUTPUTS_PATH, "weights")      
MODEL_PATH          = os.path.join(WEIGHTS_DIR, RUN_NAME + "_best_model.weights.h5")            
TRAINING_CURVE_DIR  = os.path.join(OUTPUTS_PATH, "training_curves")
TRAINING_CURVE_PATH = os.path.join(TRAINING_CURVE_DIR, RUN_NAME + "_training_curve.png")

os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(TRAINING_CURVE_DIR, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main():
    set_seed(SEED)

    # ── Data ──────────────────────────────────────────────────────────────────
    CACHE_PATH = os.path.join(DATA_PATH, "valid_image_ids.csv")

    df = pd.read_csv(TRAIN_CSV)
    print(f"Loaded {len(df)} rows from {TRAIN_CSV}")

    if not os.path.exists(CACHE_PATH):
        print("Running one-time corruption scan...")
        df = scan_valid_images(df, TRAIN_DIR)
        df.to_csv(CACHE_PATH, index=False)
    else:
        df = pd.read_csv(CACHE_PATH)
        print(f"Loaded cached valid images: {len(df)} rows")

    available_ids = set(int(f.stem) for f in Path(TRAIN_DIR).glob('*.png'))
    df = df[df['image_id'].isin(available_ids)].reset_index(drop=True)
    print(f"Filtered to {len(df)} rows with available images")

    train_df, val_df = train_test_split(df, test_size=0.15, random_state=SEED, shuffle=True)
    print(f"Train: {len(train_df)}   Val: {len(val_df)}")

    train_gen = image_data_generator(train_df, TRAIN_DIR, BATCH_SIZE, is_training=True,  resize=(IMG_W, IMG_H))
    val_gen   = image_data_generator(val_df,   TRAIN_DIR, BATCH_SIZE, is_training=False, resize=(IMG_W, IMG_H))

    steps_per_epoch  = len(train_df) // BATCH_SIZE
    validation_steps = len(val_df)   // BATCH_SIZE

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = PiCarNet(image_h=IMG_H, image_w=IMG_W,
                         dropout_first=DROPOUT_FIRST, dropout_second=DROPOUT_SECOND)
    scheduler = tf.keras.optimizers.schedules.CosineDecay(
                    initial_learning_rate=LR,
                    decay_steps=EPOCHS * steps_per_epoch,
                    alpha=1e-6)
    optimiser = tf.keras.optimizers.AdamW(learning_rate=scheduler, weight_decay=1e-4)

    # ── Train ─────────────────────────────────────────────────────────────────
    train_losses, val_losses = [], []
    best_val   = float('inf')
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_gen, optimiser, steps_per_epoch)
        val_loss   = evaluate(model, val_gen, validation_steps)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch:3d}/{EPOCHS}  "
            f"train_MSE={train_loss:.6f}  "
            f"val_MSE={val_loss:.6f}"
        )

        if val_loss < best_val:
            best_val   = val_loss
            no_improve = 0
            model.save_weights(MODEL_PATH)
            print(f"  ✓ Saved best model (val_MSE={best_val:.6f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping after {PATIENCE} epochs without improvement.")
                break

    print(f"Best val MSE: {best_val:.6f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='train')
    plt.plot(val_losses,   label='val')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title(f'Training Curve — {RUN_NAME}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(TRAINING_CURVE_PATH)
    plt.close()
    print(f"Saved training curve to {TRAINING_CURVE_PATH}")


if __name__ == "__main__":
    main()