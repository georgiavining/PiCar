import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from pathlib import Path

from picarnet import PiCarNet
from data import CarDataset, get_transforms, scan_valid_images
from train import train_one_epoch, evaluate

# ── Config ────────────────────────────────────────────────────────────────────
SEED                = 42
IMG_H               = 120
IMG_W               = 160
BATCH_SIZE          = 32
EPOCHS              = 60
LR_HEAD             = 1e-3   # head LR during frozen phase
LR_FINETUNE         = 1e-4   # full model LR after unfreezing
WEIGHT_DECAY        = 1e-3
PATIENCE            = 10
DROPOUT_FIRST_LAYER = 0.4
DROPOUT_SECOND_LAYER= 0.3
UNFREEZE_EPOCH      = 10     # epoch to unfreeze backbone
CORNER_WEIGHT       = 5.0
CORNER_THRESHOLD    = 0.10
RUN_NAME            = "finetune_mv3"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
REPO_DIR           = os.path.dirname(os.path.dirname(BASE_DIR))
DATA_PATH          = os.path.join(REPO_DIR, "data")
TRAIN_CSV          = os.path.join(DATA_PATH, "combined.csv")
TRAIN_DIR          = os.path.join(DATA_PATH, "training_images")
OUTPUTS_PATH       = os.path.join(BASE_DIR, "outputs")
MODELS_DIR         = os.path.join(OUTPUTS_PATH, "models")
MODEL_PATH         = os.path.join(MODELS_DIR, RUN_NAME + "_best_model.pth")
TRAINING_CURVE_DIR = os.path.join(OUTPUTS_PATH, "training_curves")
TRAINING_CURVE_PATH= os.path.join(TRAINING_CURVE_DIR, RUN_NAME + "_training_curve.png")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TRAINING_CURVE_DIR, exist_ok=True)


def set_seed(seed):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_sampler(train_df):
    weights = train_df['angle'].apply(
        lambda a: CORNER_WEIGHT if abs(a - 0.5) > CORNER_THRESHOLD else 1.0
    ).values
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )


def set_backbone_grad(model, requires_grad: bool):
    for param in model.backbone.parameters():
        param.requires_grad = requires_grad


def main():
    set_seed(SEED)

    # ── Data ──────────────────────────────────────────────────────────────────
    CACHE_PATH = os.path.join(DATA_PATH, "combined_valid_image_ids.csv")

    df = pd.read_csv(TRAIN_CSV)
    print(f"Loaded {len(df)} rows from {TRAIN_CSV}")
    print(df[["angle", "speed"]].describe())

    if not os.path.exists(CACHE_PATH):
        print("Running one-time corruption scan...")
        df = scan_valid_images(df, TRAIN_DIR)
        df[["image_id", "angle", "speed", "source"]].to_csv(CACHE_PATH, index=False)
    else:
        df = pd.read_csv(CACHE_PATH)
        print(f"Loaded cached valid images: {len(df)} rows")

    available_ids = set(int(f.stem) for f in Path(TRAIN_DIR).glob('*.png'))

    def row_exists(row):
        image_id = str(row['image_id'])
        if image_id.isdigit():
            return int(image_id) in available_ids
        else:
            return Path(image_id).exists()

    df = df[df.apply(row_exists, axis=1)].reset_index(drop=True)
    print(f"Filtered to {len(df)} rows with available images")

    corner_frames = (df['angle'] - 0.5).abs() > CORNER_THRESHOLD
    print(f"Corner frames: {corner_frames.sum()} / {len(df)} "
          f"({100 * corner_frames.mean():.1f}%)")

    train_df, val_df = train_test_split(
        df, test_size=0.15, random_state=SEED, shuffle=True
    )
    print(f"Train: {len(train_df)}   Val: {len(val_df)}")

    train_ds = CarDataset(train_df, TRAIN_DIR, img_h=IMG_H, img_w=IMG_W, augment=True)
    val_ds   = CarDataset(val_df,   TRAIN_DIR, img_h=IMG_H, img_w=IMG_W, augment=False)

    sampler      = make_sampler(train_df)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    # ── Model: pretrained MV3, frozen backbone ────────────────────────────────
    model = PiCarNet(
        pretrained=True,
        dropout_rate_first=DROPOUT_FIRST_LAYER,
        dropout_rate_second=DROPOUT_SECOND_LAYER,
    ).to(DEVICE)

    set_backbone_grad(model, False)
    print(f"Backbone frozen. Training head only for {UNFREEZE_EPOCH} epochs.")

    optimiser = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)
    scaler    = torch.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))

    # ── Training loop ─────────────────────────────────────────────────────────
    train_losses, val_losses = [], []
    best_val   = float('inf')
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):

        # Unfreeze backbone at UNFREEZE_EPOCH with lower LR
        if epoch == UNFREEZE_EPOCH:
            set_backbone_grad(model, True)
            optimiser = optim.AdamW(
                model.parameters(), lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimiser, T_max=EPOCHS - UNFREEZE_EPOCH
            )
            print(f"Epoch {epoch}: backbone unfrozen, LR -> {LR_FINETUNE}")

        train_loss = train_one_epoch(model, train_loader, optimiser, scaler, DEVICE)
        val_loss   = evaluate(model, val_loader, DEVICE)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"train_MSE={train_loss:.6f}  val_MSE={val_loss:.6f}  lr={lr:.2e}")

        if val_loss < best_val:
            best_val   = val_loss
            no_improve = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  ✓ Saved best model (val_MSE={best_val:.6f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping after {PATIENCE} epochs without improvement.")
                break

    print(f"Best validation MSE: {best_val:.6f}")

    # ── Training curve ────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='train')
    plt.plot(val_losses,   label='val')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title(f'Training Curve — {RUN_NAME}')
    plt.legend()
    plt.savefig(TRAINING_CURVE_PATH)
    plt.close()
    print(f"Training curve saved to {TRAINING_CURVE_PATH}")


if __name__ == "__main__":
    main()