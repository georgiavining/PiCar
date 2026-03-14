import tensorflow as tf
import os
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

def weighted_bce(y_true, y_pred, weight_for_1=0.6, weight_for_0=2.4):
    """Weighted binary cross-entropy for imbalanced speed labels"""
    weights = y_true * weight_for_1 + (1 - y_true) * weight_for_0
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(weights * bce)

# ── W&B Callback ──────────────────────────────────────────────────────────────
 
class MetricPrefixLogger(tf.keras.callbacks.Callback):
    """Logs Keras epoch metrics to W&B with a prefix and optional step offset.
 
    Args:
        prefix:      String prepended to every metric name (e.g. "angle", "speed").
        step_offset: Added to the epoch number so angle and speed steps don't
                     collide on the same W&B x-axis.
    """
 
    def __init__(self, prefix: str, step_offset: int = 0):
        super().__init__()
        self.prefix = prefix
        self.step_offset = step_offset
 
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            wandb.log(
                {f"{self.prefix}_{k}": v for k, v in logs.items()},
                step=self.step_offset + epoch,
                commit=True,
            )
 
 
# ── Training ──────────────────────────────────────────────────────────────────
 
def train_models(
    angle_model,
    speed_model,
    angle_train,
    angle_val,
    speed_train,
    speed_val,
    models_path: str,
    config: dict,
    wandb_log: bool = True,
    angle_loss="mse",
    speed_loss=weighted_bce,
):
    tf.keras.backend.clear_session()
 
    hp          = config.get("hyperparameters", {})
    lr          = hp.get("learning_rate", 1e-3)
    angle_epochs = hp.get("angle_epochs", 50)
    speed_epochs = hp.get("speed_epochs", 100)
    model_name  = config.get("experiment_name")
 
    # ── Compile ───────────────────────────────────────────────────────────────
 
    angle_model.compile(
        loss=angle_loss,
        optimizer=tf.keras.optimizers.Adam(lr),
        metrics=["mse"],
    )
 
    speed_model.compile(
        loss=speed_loss,
        optimizer=tf.keras.optimizers.Adam(lr),
        metrics=["mse"],
    )
 
    # ── Shared callbacks ──────────────────────────────────────────────────────────
 
    def common_callbacks(checkpoint_path):
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_mse", patience=7, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_mse", factor=0.5, patience=3, min_lr=1e-6
            ),
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path, monitor="val_mse", save_best_only=True
            ),
        ]
 
    angle_callbacks = common_callbacks(
        os.path.join(models_path, f"{model_name}_angle_model_best.keras")
    )
    speed_callbacks = common_callbacks(
        os.path.join(models_path, f"{model_name}_speed_model_best.keras")
    )
 
    history_angle = angle_model.fit(
        angle_train,
        validation_data=angle_val,
        epochs=angle_epochs,
        callbacks=angle_callbacks,
        verbose=1,
    )
 
    history_speed = speed_model.fit(
        speed_train,
        validation_data=speed_val,
        epochs=speed_epochs,
        callbacks=speed_callbacks,
        verbose=1,
    )
 
    return angle_model, speed_model, history_angle, history_speed
 