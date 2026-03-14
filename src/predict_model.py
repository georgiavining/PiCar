import tensorflow as tf
import os
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint, WandbEval
import numpy as np
import pandas as pd

def generate_test_predictions(angle_model, speed_model, test_ds,
                                        predictions_path, model_name):

    angle_preds, speed_preds = [], []

    for x in test_ds:
        angle_preds.extend(angle_model.predict(x, verbose=0).flatten())
        speed_preds.extend((speed_model.predict(x, verbose=0) > 0.5).astype(int).flatten())

    df_preds = pd.DataFrame({
        "image_id": range(len(angle_preds)),
        "angle": np.array(angle_preds),
        "speed": np.array(speed_preds)
    })

    os.makedirs(predictions_path, exist_ok=True)
    csv_output = os.path.join(predictions_path, f"{model_name}.csv")
    df_preds.to_csv(csv_output, index=False)
    print(f"Saved to: {csv_output}")
    return df_preds