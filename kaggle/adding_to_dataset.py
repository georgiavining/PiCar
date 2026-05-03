import os
import pandas as pd
import shutil
from pathlib import Path

BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
REPO_DIR          = os.path.dirname(BASE_DIR)
OUTPUTS_PATH    = os.path.join(BASE_DIR, "outputs")
DATA_PATH         = os.path.join(REPO_DIR, "data")
TRAIN_CSV         = os.path.join(DATA_PATH, "train.csv")
TRAIN_DIR  = Path(os.path.join(REPO_DIR, "data", "training_images"))
CACHE_PATH = Path(os.path.join(DATA_PATH, "valid_image_ids.csv"))
NEW_IMG_DIR = Path(os.path.join(REPO_DIR, "data", "new_data"))


#-- load existing data -------------------------------------------
original_df = pd.read_csv(TRAIN_CSV)
print(f"Original dataset: {len(original_df)} rows")

#-- find next available image id ---------------------------------
existing_ids = set(int(f.stem) for f in TRAIN_DIR.glob("*.png"))
next_id      = max(existing_ids) + 1
print(f"Starting new IDs from: {next_id}")

#-- parse, rename and copy new images ----------------------------
records = []

for f in sorted(NEW_IMG_DIR.glob("*.png")):
    parts = f.stem.split("_")
    try:
        angle = float(parts[1])
        speed = float(parts[2])
    except (IndexError, ValueError):
        print(f"Skipping malformed filename: {f.name}")
        continue

    new_name = f"{next_id}.png"
    dest     = TRAIN_DIR / new_name
    shutil.copy(f, dest)

    records.append({
        "image_id": next_id,
        "angle":    angle,
        "speed":    speed
    })
    next_id += 1

print(f"Copied and renamed {len(records)} new images")

#-- combine and save ---------------------------------------------
new_df      = pd.DataFrame(records)
new_df_oversampled = pd.concat([new_df] * 5, ignore_index=True)
combined_df = pd.concat([original_df, new_df_oversampled], ignore_index=True)
combined_df.to_csv(TRAIN_CSV, index=False)
print(f"Combined dataset: {len(combined_df)} rows saved to {TRAIN_CSV}")

#-- delete cache so it rescans -----------------------------------
if CACHE_PATH.exists():
    CACHE_PATH.unlink()
    print("Deleted image cache — will rescan on next training run")

print(f"\nSummary:")
print(f"  Original images:        {len(original_df)}")
print(f"  New images copied:      {len(records)}")
print(f"  New images oversampled: {len(new_df_oversampled)}")
print(f"  Combined dataset:       {len(combined_df)}")