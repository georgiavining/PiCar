import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from lane_model import PiCarNet
from data import scan_valid_images, make_tf_dataset
from train import train_one_epoch, evaluate


SEED = 42
IMG_H = 260
IMG_W = 260
BATCH_SIZE = 32
EPOCHS = 60
LR = 1e-4
PATIENCE = 10
RUN_NAME = "mobilenet_angle_debug"

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.dirname(CODE_DIR)
REPO_DIR = os.path.dirname(os.path.dirname(MODEL_DIR))

DATA_PATH = os.path.join(REPO_DIR, "kaggle_data")
TRAIN_CSV = os.path.join(DATA_PATH, "train.csv")
TRAIN_DIR = os.path.join(DATA_PATH, "training_images")

OUTPUTS_PATH = os.path.join(MODEL_DIR, "outputs")
WEIGHTS_DIR = os.path.join(OUTPUTS_PATH, "weights")
MODEL_PATH = os.path.join(WEIGHTS_DIR, RUN_NAME + "_best.weights.h5")

CURVE_DIR = os.path.join(OUTPUTS_PATH, "training_curves")
CURVE_PATH = os.path.join(CURVE_DIR, RUN_NAME + ".png")

os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(CURVE_DIR, exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def main():
    set_seed(SEED)

    df = pd.read_csv(TRAIN_CSV)
    df = scan_valid_images(df, TRAIN_DIR)

    train_df, val_df = train_test_split(
        df, test_size=0.15, random_state=SEED, shuffle=True
    )

    train_ds = make_tf_dataset(
        train_df,
        TRAIN_DIR,
        BATCH_SIZE,
        is_training=True,
        resize=(IMG_H, IMG_W)
    )

    val_ds = make_tf_dataset(
        val_df,
        TRAIN_DIR,
        BATCH_SIZE,
        is_training=False,
        resize=(IMG_H, IMG_W)
    )

    model = PiCarNet(image_h=IMG_H, image_w=IMG_W)

    for imgs, _ in train_ds.take(1):
        model(imgs)

    model.backbone.trainable = False

    optimiser = tf.keras.optimizers.Adam(LR)

    train_losses, val_losses = [], []
    best_val = float("inf")
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):

        if epoch == 1:
            for imgs, angles in train_ds.take(1):
                preds = model(imgs, training=False)
                print("pred mean:", tf.reduce_mean(preds).numpy())
                print("angle mean:", tf.reduce_mean(angles).numpy())
                print("pred std:", tf.math.reduce_std(preds).numpy())
                print("angle std:", tf.math.reduce_std(angles).numpy())

        train_loss = train_one_epoch(model, train_ds, optimiser)
        val_loss = evaluate(model, val_ds)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train={train_loss:.5f} | val={val_loss:.5f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            model.save_weights(MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

        if epoch % 5 == 0:
            for imgs, angles in train_ds.take(1):
                preds = model(imgs, training=False)
                print("pred mean:", tf.reduce_mean(preds).numpy())
                print("pred std:", tf.math.reduce_std(preds).numpy())

    print(f"Best val loss: {best_val:.5f}")

    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.title(RUN_NAME)
    plt.savefig(CURVE_PATH)
    plt.close()

if __name__ == "__main__":
    main()