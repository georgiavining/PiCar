import os, random, datetime
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import pandas as pd

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
from sklearn.model_selection import train_test_split

from driver.lane_navigation.code.data import scan_valid_images, make_tf_dataset
from driver.lane_navigation.code.seed import set_seed
from driver.lane_navigation.code.mv2model import create_mv2_model

#--Paths----------------------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parent
LANE_NAV_DIR = CODE_DIR.parent
OUTPUTS_DIR = LANE_NAV_DIR / "outputs"
WEIGHTS_DIR = OUTPUTS_DIR / "weights"
TRAINING_CURVE_DIR = OUTPUTS_DIR / "training_curves"

REPO_DIR = Path(__file__).resolve().parents[4]
DATA_DIR = REPO_DIR / "data"
TRAIN_CSV = DATA_DIR / "train.csv"
TRAIN_DIR = DATA_DIR / "training_images"
CACHE_DIR = DATA_DIR / "valid_image_ids.csv"

#--Config----------------------------------------------------------------------------
HEIGHT = 224
WIDTH = 224
BATCH_SIZE = 32
LR = 1e-4
PATIENCE=10
RUN_NAME = "mv2_run1"
SEED = 42

#--Main------------------------------------------------------------------------------
def main():
    set_seed(SEED)
    mixed_precision.set_global_policy(policy="mixed_float16")
    print(f"Using TensorFlow version: {tf.__version__}")
    print(f"Using GPU: {tf.config.list_physical_devices('GPU')}")
    print(f"Using mixed precision: {mixed_precision.global_policy()}")
    
    # ── Data ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(TRAIN_CSV)

    if not os.path.exists(CACHE_DIR):
        print("Running one-time corruption scan (this may take a minute)...")
        df = scan_valid_images(df, TRAIN_DIR)
        df.to_csv(CACHE_DIR, index=False)
        print(f"Saved valid image cache to {CACHE_DIR}")
    else:
        df = pd.read_csv(CACHE_DIR)

    train_df, val_df = train_test_split(df, test_size=0.15, random_state=SEED)

    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mv2_preprocess
    train_ds = make_tf_dataset(train_df, TRAIN_DIR, BATCH_SIZE, is_training=True, resize=(HEIGHT, WIDTH), preprocess_fn=mv2_preprocess)
    print(f"Training dataset: {len(train_df)} samples")

    val_ds   = make_tf_dataset(val_df, TRAIN_DIR, BATCH_SIZE, is_training=False, resize=(HEIGHT, WIDTH),  preprocess_fn=mv2_preprocess)
    print(f"Validation dataset: {len(val_df)} samples")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = create_mv2_model(input_shape=(HEIGHT, WIDTH, 3))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss='mse',
        metrics=['mae']
    )

    # ── Callbacks ───────────────────────────────────────────────────────────────
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        WEIGHTS_DIR / f"{RUN_NAME}_best_model.h5",
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    )
    
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        verbose=1,
        min_lr=1e-6
        )

    # ── Train ─────────────────────────────────────────────────────────────────
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=100,
        callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr]
    )
    # ── Save training curve ─────────────────────────────────────────────────────
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(TRAINING_CURVE_DIR / f"{RUN_NAME}_training_curve.png")
    print(f"Training complete! Best model saved to {WEIGHTS_DIR / f'{RUN_NAME}_best_model.h5'}")

if __name__ == "__main__":
    main()