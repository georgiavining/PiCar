import yaml
import wandb
import random
import numpy as np
import tensorflow as tf
import pandas as pd
import os
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
from src.build_model import build_model_from_config
from src.train_model import train_models, weighted_bce
from src.build_dataset import build_dataset, split_indices, remove_invalid_images
from src.seed import set_seed

wandb.login(key="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")

BASE_DIR          = os.path.dirname(os.path.abspath(__file__))

TRAIN_CSV         = os.path.join(BASE_DIR, "data", "train.csv")
TRAIN_IMAGES      = os.path.join(BASE_DIR, "data", "training_images")
TEST_IMAGES       = os.path.join(BASE_DIR, "data", "test_images")
MODELS_PATH       = os.path.join(BASE_DIR, "output", "models")
PREDICTIONS_PATH  = os.path.join(BASE_DIR, "output", "predictions")

os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(PREDICTIONS_PATH, exist_ok=True)

SEED = 42
TEST_SPLIT = 0.1
VAL_SPLIT = 0.1
IMG_SIZE = 224
BATCH_SIZE = 8

def main():
    set_seed(SEED)
 
    # ── Data ──────────────────────────────────────────────────────────────────
 
    df = pd.read_csv(TRAIN_CSV)
    df, _ = remove_invalid_images(df, TRAIN_IMAGES)
    df = df[:100] 

    angles = df["angle"].values
    speeds = df["speed"].values
    filenames = np.array(
        [os.path.join(TRAIN_IMAGES, f"{i}.png") for i in df["image_id"]]
    )
 
    train_idx, val_idx, test_idx = split_indices(
        len(filenames), val_split=VAL_SPLIT, test_split=TEST_SPLIT
    )
 
    def build_splits(targets):
        return (
            build_dataset(filenames[train_idx], targets[train_idx], img_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True),
            build_dataset(filenames[val_idx],   targets[val_idx],   img_size=IMG_SIZE, batch_size=BATCH_SIZE),
            build_dataset(filenames[test_idx],  targets[test_idx],  img_size=IMG_SIZE, batch_size=BATCH_SIZE),
        )
 
    angle_train, angle_val, angle_test = build_splits(angles)
    speed_train, speed_val, speed_test = build_splits(speeds)
 
    # ── Config & W&B ──────────────────────────────────────────────────────────
 
    with open("configs/experiment_1.yaml") as f:
        exp_cfg = yaml.safe_load(f)
 
    wandb.init(
        entity=exp_cfg["wandb_entity"],
        project=exp_cfg["project_name"],
        name=exp_cfg["experiment_name"],
        config=exp_cfg,
    )
 
    # ── Models ────────────────────────────────────────────────────────────────
 
    angle_model = build_model_from_config(exp_cfg["angle_model"])
    speed_model = build_model_from_config(exp_cfg["speed_model"])
 
    # ── Train ─────────────────────────────────────────────────────────────────
 
    angle_model, speed_model, hist_angle, hist_speed = train_models(
        angle_model,
        speed_model,
        angle_train, angle_val,
        speed_train, speed_val,
        models_path=MODELS_PATH,
        config=exp_cfg,
        wandb_log=True,
        angle_loss="mse",
        speed_loss=weighted_bce,
    )
 
    # ── Evaluate ──────────────────────────────────────────────────────────────
 
    _, angle_test_mse = angle_model.evaluate(angle_test, verbose=0)
    _, speed_test_mse = speed_model.evaluate(speed_test, verbose=0)

    speed_preds = speed_model.predict(speed_test)
    print(
        f"Speed preds  min={speed_preds.min():.4f} "
        f"max={speed_preds.max():.4f} "
        f"mean={speed_preds.mean():.4f} "
        f"std={speed_preds.std():.4f}"
    )
 
    wandb.log({
        "angle_test_mse":   angle_test_mse,
        "speed_test_mse":   speed_test_mse,
        "speed_pred_mean":  float(speed_preds.mean()),
        "speed_pred_std":   float(speed_preds.std()),
    }, commit=True)
 
    wandb.finish()
 
 
if __name__ == "__main__":
    main()