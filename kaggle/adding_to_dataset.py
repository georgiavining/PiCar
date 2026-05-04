import os
import pandas as pd
import shutil
from pathlib import Path

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
REPO_DIR        = os.path.dirname(BASE_DIR)
OUTPUTS_PATH    = os.path.join(BASE_DIR, "outputs")
DATA_PATH       = os.path.join(REPO_DIR, "data")
TRAIN_CSV       = os.path.join(DATA_PATH, "train.csv")
TRAIN_DIR       = Path(os.path.join(REPO_DIR, "data", "training_images"))
CACHE_PATH      = Path(os.path.join(DATA_PATH, "valid_image_ids.csv"))
NEW_IMG_DIR     = Path(os.path.join(REPO_DIR, "data", "new_data"))

OVERSAMPLE_FACTOR = 1


# =========================================================
# LOAD ORIGINAL DATA
# =========================================================
full_df = pd.read_csv(TRAIN_CSV)
original_df = full_df[full_df['image_id'] <= 14418].reset_index(drop=True)
print(f"Original dataset: {len(original_df)} rows")


# =========================================================
# REMOVE OLD NEW DATA IMAGES
# =========================================================
print("Removing previously copied new images...")
removed = 0

for f in TRAIN_DIR.glob("*.png"):
    if f.stem.isdigit() and int(f.stem) > 14418:
        f.unlink()
        removed += 1

print(f"Removed {removed} old new-data images")


# =========================================================
# COPY + PARSE NEW DATA
# =========================================================
next_id = 14419
records = []

for scenario_folder in NEW_IMG_DIR.iterdir():
    if not scenario_folder.is_dir():
        continue

    scenario = scenario_folder.name

    for f in sorted(scenario_folder.glob("*.png")):
        parts = f.stem.split("_")

        try:
            angle = float(parts[1])
            speed = float(parts[2])
        except:
            continue

        new_name = f"{next_id}.png"
        shutil.copy(f, TRAIN_DIR / new_name)

        records.append({
            "image_id": next_id,
            "angle": (angle - 50) / 80,
            "speed": speed / 35,
            "scenario": scenario
        })

        next_id += 1

new_df = pd.DataFrame(records)

# =========================================================
# SCENARIO BALANCING
# =========================================================
figure8       = new_df[new_df["scenario"] == "figure8"]
figure8_obj    = new_df[new_df["scenario"] == "figure8_object"]
oval_inner     = new_df[new_df["scenario"] == "oval_inner"]
oval_outer     = new_df[new_df["scenario"] == "oval_outer"]
t_left         = new_df[new_df["scenario"] == "tjunc_left"]
t_right        = new_df[new_df["scenario"] == "tjunc_right"]

figure8 = figure8.sample(frac=0.5, random_state=42)        # reduce aggressive turning
oval_inner = oval_inner.sample(frac=0.3, random_state=42)  # reduce tight loop bias

oval_outer = oval_outer
t_left = t_left
t_right = t_right
figure8_obj = figure8_obj

balanced_new_df = pd.concat([
    figure8,
    figure8_obj,
    oval_inner,
    oval_outer,
    t_left,
    t_right
]).reset_index(drop=True)


# =========================================================
# GLOBAL STEERING BALANCE (VERY IMPORTANT)
# prevents circling / directional drift
# =========================================================
left  = balanced_new_df[balanced_new_df['angle'] < 0.5]
right = balanced_new_df[balanced_new_df['angle'] > 0.5]

n = min(len(left), len(right))

if n > 0:
    balanced_new_df = pd.concat([
        left.sample(n=n, random_state=42),
        right.sample(n=n, random_state=42)
    ]).reset_index(drop=True)


# =========================================================
# OPTIONAL OVERSAMPLING
# =========================================================
if OVERSAMPLE_FACTOR > 1:
    balanced_new_df = pd.concat(
        [balanced_new_df] * OVERSAMPLE_FACTOR,
        ignore_index=True
    )


# =========================================================
# COMBINE WITH ORIGINAL (ALL KEPT)
# =========================================================
combined_df = pd.concat(
    [original_df, balanced_new_df],
    ignore_index=True
)

combined_df.to_csv(TRAIN_CSV, index=False)

print(f"\nCombined dataset saved to {TRAIN_CSV}")


# =========================================================
# CLEAN CACHE
# =========================================================
if CACHE_PATH.exists():
    CACHE_PATH.unlink()
    print("Deleted image cache — will rescan on next training run")


# =========================================================
# SUMMARY
# =========================================================
print("\nSummary:")
print(f"  Original images:   {len(original_df)}")
print(f"  New images:        {len(records)}")
print(f"  New used (final):  {len(balanced_new_df)}")
print(f"  Combined dataset:  {len(combined_df)}")