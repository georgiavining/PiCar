import tensorflow as tf
import os
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

def weighted_bce(y_true, y_pred, weight_for_1=0.6, weight_for_0=2.4):
    """Weighted binary cross-entropy for imbalanced speed labels"""
    weights = y_true * weight_for_1 + (1 - y_true) * weight_for_0
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(weights * bce)

class MetricPrefixLogger(tf.keras.callbacks.Callback):

    def __init__(self, prefix):
        super().__init__()
        self.prefix = prefix

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            prefixed_logs = {f"{self.prefix}_{k}": v for k, v in logs.items()}
            wandb.log(prefixed_logs, step=epoch)

def train_models(
    angle_model,
    speed_model,
    angle_train,
    angle_val,
    speed_train,
    speed_val,
    models_path,
    config,
    wandb_log=True,
    angle_loss="mse",
    speed_loss=weighted_bce
):

    tf.keras.backend.clear_session()

    hp = config.get("hyperparameters", {})
    lr = hp.get("learning_rate", 1e-3)
    angle_epochs = hp.get("angle_epochs", 50)
    speed_epochs = hp.get("speed_epochs", 100)
    model_name = config.get("experiment_name")

    angle_model.compile(
        loss=angle_loss,
        optimizer=tf.keras.optimizers.Adam(lr),
        metrics=["mse"]
    )

    speed_model.compile(
        loss=speed_loss,
        optimizer=tf.keras.optimizers.Adam(lr),
        metrics=["mse"]
    )

    common_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_mse", patience=7, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_mse", factor=0.5, patience=3, min_lr=1e-6
        )
    ]

    angle_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(models_path, f"{model_name}_angle_model_best.keras"),
        monitor="val_mse",
        save_best_only=True
    )

    speed_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(models_path, f"{model_name}_speed_model_best.keras"),
        monitor="val_mse",
        save_best_only=True
    )

    # Metric prefix loggers
    angle_prefix_cb = MetricPrefixLogger("angle")
    speed_prefix_cb = MetricPrefixLogger("speed")

    # W&B callbacks (optional)
    if wandb_log:
        angle_wandb_callbacks = [
            WandbMetricsLogger(),
            WandbModelCheckpoint(
                filepath=os.path.join(models_path, "angle_model_best.keras"),
                monitor="val_mse",
                save_best_only=True,
                mode="min"
            )
        ]

        speed_wandb_callbacks = [
            WandbMetricsLogger(),
            WandbModelCheckpoint(
                filepath=os.path.join(models_path, "speed_model_best.keras"),
                monitor="val_mse",
                save_best_only=True,
                mode="min"
            )
        ]
    else:
        angle_wandb_callbacks = []
        speed_wandb_callbacks = []

    history_angle = angle_model.fit(
        angle_train,
        validation_data=angle_val,
        epochs=angle_epochs,
        callbacks=common_callbacks + [angle_checkpoint, angle_prefix_cb] + angle_wandb_callbacks,
        verbose=1
    )

    history_speed = speed_model.fit(
        speed_train,
        validation_data=speed_val,
        epochs=speed_epochs,
        callbacks=common_callbacks + [speed_checkpoint, speed_prefix_cb] + speed_wandb_callbacks,
        verbose=1
    )

    return angle_model, speed_model, history_angle, history_speed