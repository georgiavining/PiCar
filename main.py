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

TRAIN_CSV = "data/train.csv"
TRAIN_IMAGES = "data/training_images"
TEST_IMAGES = "data/test_images"
MODELS_PATH = "output/models/"
PREDICTIONS_PATH = "output/predictions/"

SEED = 42
TEST_SPLIT = 0.1
VAL_SPLIT = 0.1
IMG_SIZE = 224
BATCH_SIZE = 8

set_seed(SEED)

df = pd.read_csv(TRAIN_CSV)
df, _ = remove_invalid_images(df, TRAIN_IMAGES)
df = df[:100]  # For debugging, use only a subset of the data
angles = df['angle'].values
speeds = df['speed'].values

filenames = [os.path.join(TRAIN_IMAGES, f"{i}.png") for i in df['image_id']]
fn = np.array(filenames)

train_idx, val_idx, test_idx = split_indices(len(filenames), val_split=VAL_SPLIT, test_split=TEST_SPLIT)

def build_train_val_test(fn, targets):
    return (
        build_dataset(fn[train_idx], targets[train_idx], img_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True),
        build_dataset(fn[val_idx],   targets[val_idx],   img_size=IMG_SIZE, batch_size=BATCH_SIZE),
        build_dataset(fn[test_idx],  targets[test_idx],  img_size=IMG_SIZE, batch_size=BATCH_SIZE)
    )

angle_train, angle_val, angle_test = build_train_val_test(fn, angles)
speed_train, speed_val, speed_test = build_train_val_test(fn, speeds)

with open("configs/experiment_1.yaml", "r") as f:
    exp_cfg = yaml.safe_load(f)

angle_model = build_model_from_config(exp_cfg["angle_model"])
speed_model = build_model_from_config(exp_cfg["speed_model"])

wandb.init(
    entity=exp_cfg["wandb_entity"],
    project=exp_cfg["project_name"],
    name=exp_cfg["experiment_name"],
    config=exp_cfg
)

angle_model, speed_model, hist_angle, hist_speed = train_models(
    angle_model,
    speed_model,
    angle_train,
    angle_val,
    speed_train,
    speed_val,
    models_path=MODELS_PATH,
    config=exp_cfg,
    wandb_log=True,
    angle_loss = "mse",
    speed_loss = weighted_bce
)

angle_test_loss, angle_test_mse = angle_model.evaluate(angle_test)
speed_test_loss, speed_test_mse = speed_model.evaluate(speed_test)

wandb.log({
    "angle_test_mse": angle_test_mse,
    "speed_test_mse": speed_test_mse,
})

wandb.finish()