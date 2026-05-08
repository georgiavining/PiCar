import os
import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
REPO_DIR     = os.path.dirname(os.path.dirname(BASE_DIR))
DATA_DIR   = os.path.join(REPO_DIR, "data")
TRAIN_CSV    = os.path.join(DATA_DIR, "train.csv")        
COMBINED_CSV = os.path.join(DATA_DIR, "combined.csv")    
CACHE_PATH   = Path(os.path.join(DATA_DIR, "combined_valid_ids.csv"))
NEW_IMG_DIR  = Path(os.path.join(DATA_DIR, "collected_images"))

# ── Config ────────────────────────────────────────────────────────────────────
# Comment out subsets you don't want to include in this training run
SUBSETS = [
    "oval_inner",
    "oval_outer",
    "figure8",
    "figure8_object",
    "tjunc_left",
    "tjunc_right",
]
OVERSAMPLE_FACTOR = 1   # set >1 to repeat new data rows in CSV (no file copying)

# ── Normalisation (must match model.py predict()) ─────────────────────────────
ANGLE_MIN = 50
ANGLE_MAX = 130
SPEED_MAX = 35

def normalise_angle(raw):
    return (raw - 50) / (ANGLE_MAX - ANGLE_MIN)

def normalise_speed(raw):
    return raw / SPEED_MAX


def main():

    # ── Load original data ────────────────────────────────────────────────────
    original_df = pd.read_csv(TRAIN_CSV)
    original_df = original_df.copy()
    original_df['source'] = 'original'
    print(f"Original dataset: {len(original_df)} rows")

    # ── Parse new data ────────────────────────────────────────────────────────
    # Walk each selected subset folder. Filenames are <TIMESTAMP>_<ANGLE>_<SPEED>.png.
    # We store the full absolute path as image_id so CarDataset can find the
    # file without any copying. Angle and speed are normalised to [0, 1].
    records = []

    for subset in SUBSETS:
        subset_dir = NEW_IMG_DIR / subset
        if not subset_dir.exists():
            print(f"WARNING: {subset_dir} not found, skipping")
            continue

        subset_records = []
        for f in sorted(subset_dir.glob("*.png")):
            parts = f.stem.split("_")
            try:
                raw_angle = float(parts[1])
                raw_speed = float(parts[2])
            except (IndexError, ValueError):
                print(f"Skipping malformed filename: {f.name}")
                continue

            subset_records.append({
                "image_id": str(f.resolve()),   # full absolute path
                "angle"   : normalise_angle(raw_angle),
                "speed"   : normalise_speed(raw_speed),
                "source"  : subset,
            })

        print(f"  {subset}: {len(subset_records)} images found")
        records.extend(subset_records)

    print(f"Total new images: {len(records)}")

    # ── Oversample new data (optional) ────────────────────────────────────────
    # Repeats new data rows in the CSV without touching any files.
    # Useful if new data is small relative to the 14k original.
    new_df = pd.DataFrame(records)
    if OVERSAMPLE_FACTOR > 1:
        new_df = pd.concat(
            [new_df] * OVERSAMPLE_FACTOR, ignore_index=True
        )
        print(f"New data oversampled {OVERSAMPLE_FACTOR}x: {len(new_df)} rows")

    # ── Combine and save ──────────────────────────────────────────────────────
    # Original rows always kept in full. New rows appended after.
    # Saved to combined.csv — train.csv is never modified.
    combined_df = pd.concat(
        [original_df[["image_id", "angle", "speed", "source"]],
         new_df],
        ignore_index=True
    )
    combined_df.to_csv(COMBINED_CSV, index=False)
    print(f"Saved combined CSV: {COMBINED_CSV}")

    # ── Delete combined cache ─────────────────────────────────────────────────
    # main.py caches valid image IDs to avoid rescanning every run.
    # Delete it so it gets rebuilt with the new image paths on next training run.
    if CACHE_PATH.exists():
        CACHE_PATH.unlink()
        print("Deleted combined image cache — will rescan on next training run")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\nSummary:")
    print(f"  Original images  : {len(original_df)}")
    print(f"  New images (raw) : {len(records)}")
    print(f"  New images (used): {len(new_df)}")
    print(f"  Combined total   : {len(combined_df)}")
    print(f"  Active subsets   : {SUBSETS}")
    print(f"\nAngle distribution:")
    print(combined_df["angle"].describe())
    corner = (combined_df["angle"] - 0.5).abs() > 0.10
    print(f"\nCorner frames: {corner.sum()} / {len(combined_df)} "
          f"({100 * corner.mean():.1f}%)")


if __name__ == "__main__":
    main()