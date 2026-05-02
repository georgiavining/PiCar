import os, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from data import scan_valid_images, get_transforms, CarDataset
from seed import set_seed
from picarnet import PiCarNet
from train import train_one_epoch, evaluate


#--Config----------------------------------------------------------------------------

SEED         = 42
IMG_H        = 120
IMG_W        = 160 
BATCH_SIZE   = 32
EPOCHS       = 20
LR           = 5e-5
WEIGHT_DECAY = 1e-3
PATIENCE     = 7
DROPOUT_FIRST_LAYER = 0.4
DROPOUT_SECOND_LAYER = 0.3
RUN_NAME = "v2"
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#--Paths----------------------------------------------------------------------------
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))

DATA_PATH         = os.path.join(BASE_DIR, "data")
TRAIN_CSV         = os.path.join(DATA_PATH, "train.csv")
TRAIN_DIR      = os.path.join(DATA_PATH, "training_images")
TEST_DIR       = os.path.join(DATA_PATH, "test_images")
OUTPUTS_PATH    = os.path.join(BASE_DIR, "kaggle_outputs")
MODELS_DIR = os.path.join(OUTPUTS_PATH, "models")
MODEL_PATH = os.path.join(MODELS_DIR, RUN_NAME+"_best_model.pth")
SAVED_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pth")
PREDICTIONS_DIR = os.path.join(OUTPUTS_PATH, "predictions")
PREDICTIONS_PATH = os.path.join(PREDICTIONS_DIR, RUN_NAME+".csv")
TRAINING_CURVE_DIR = os.path.join(OUTPUTS_PATH, "training_curves")
TRAINING_CURVE_PATH = os.path.join(TRAINING_CURVE_DIR, RUN_NAME + '_training_curve.png')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(TRAINING_CURVE_DIR, exist_ok=True)

#--Main------------------------------------------------------------------------------
def main():
    set_seed(SEED)
 
    # ── Data ──────────────────────────────────────────────────────────────────
 
    CACHE_PATH = os.path.join(DATA_PATH, "valid_image_ids.csv")

    df = pd.read_csv(TRAIN_CSV)
    print(f"Loaded {len(df)} rows from {TRAIN_CSV}")
    print(df[["angle", "speed"]].describe())

    if not os.path.exists(CACHE_PATH):
        print("Running one-time corruption scan (this may take a minute)...")
        df = scan_valid_images(df, TRAIN_DIR)
        df.to_csv(CACHE_PATH, index=False)
        print(f"Saved valid image cache to {CACHE_PATH}")
    else:
        df = pd.read_csv(CACHE_PATH)
        print(f"Loaded cached valid images: {len(df)} rows")
    
    available_ids = set(int(f.stem) for f in Path(TRAIN_DIR).glob('*.png'))
    df = df[df['image_id'].isin(available_ids)].reset_index(drop=True)
    print(f"Filtered to {len(df)} rows with available images")

    train_df, val_df = train_test_split(df, test_size=0.15, random_state=SEED, shuffle=True)
    print(f'Train: {len(train_df)}   Val: {len(val_df)}')

    train_ds = CarDataset(train_df, TRAIN_DIR, transform=get_transforms(augment=True,img_h=IMG_H, img_w=IMG_W))
    val_ds   = CarDataset(val_df,   TRAIN_DIR, transform=get_transforms(augment=False,img_h=IMG_H, img_w=IMG_W))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    print('Data loaders ready!')

 
    # ── Model ────────────────────────────────────────────────────────────────
 
    model = PiCarNet(pretrained=False,dropout_rate_first=DROPOUT_FIRST_LAYER,dropout_rate_second=DROPOUT_SECOND_LAYER).to(DEVICE)
    model.load_state_dict(torch.load(SAVED_MODEL_PATH, map_location=DEVICE))
    optimiser = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)
    scaler    = torch.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))
 
    # ── Train ─────────────────────────────────────────────────────────────────
 
    train_losses = []
    val_losses   = []

    best_val   = float('inf')
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimiser, scaler, DEVICE)
        val_loss   = evaluate(model, val_loader, DEVICE)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        lr = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch:3d}/{EPOCHS}  train_MSE={train_loss:.6f}  val_MSE={val_loss:.6f}  lr={lr: .2e}')
        
        if val_loss < best_val:
            best_val   = val_loss
            no_improve = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f'  checkmark Saved best model (val_MSE={best_val:.6f})')
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f'  Early stopping after {PATIENCE} epochs without improvement.')
                break

    print(f'Best validation MSE: {best_val:.6f}')
    #-- Plotting loss curves (optional) --
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='train')
    plt.plot(val_losses,   label='val')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Training Curve')
    plt.legend()
    plt.savefig(TRAINING_CURVE_PATH)
    plt.close()


if __name__ == "__main__":
    main()