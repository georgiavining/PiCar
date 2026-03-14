import tensorflow as tf
import os
import wandb

def build_model_from_config(cfg, augment_cfg=None, name_prefix="model"):

    inputs = tf.keras.layers.Input(shape=tuple(cfg["input_shape"]))
    x = inputs


    if augment_cfg:
        for aug_type, params in augment_cfg.items():
            if aug_type == "flip" and params.get("mode", "horizontal") == "horizontal":
                x = tf.keras.layers.RandomFlip("horizontal", name=f"{name_prefix}_flip")(x)
            elif aug_type == "rotation":
                x = tf.keras.layers.RandomRotation(params.get("factor", 0.1), name=f"{name_prefix}_rot")(x)
            elif aug_type == "zoom":
                x = tf.keras.layers.RandomZoom(params.get("height_factor", 0.0),
                                            params.get("width_factor", 0.0),
                                            name=f"{name_prefix}_zoom")(x)
            elif aug_type == "contrast":
                x = tf.keras.layers.RandomContrast(params.get("factor", 0.1), name=f"{name_prefix}_contrast")(x)
            elif aug_type == "brightness":
                factor = params.get("factor", 0.1)
                x = tf.keras.layers.RandomBrightness(factor, name=f"{name_prefix}_brightness")(x)
    
    for i, block in enumerate(cfg.get("conv_blocks", [])):
        x = tf.keras.layers.Conv2D(
            block["filters"], 
            tuple(block.get("kernel_size", [3, 3])),
            padding="same",
            name=f"{name_prefix}_conv{i+1}"
        )(x)
        if block.get("batch_norm", False):
            x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn{i+1}")(x)
        if "activation" in block:
            x = tf.keras.layers.Activation(block["activation"], name=f"{name_prefix}_act{i+1}")(x)
        pool_type = block.get("pool", None)
        if pool_type == "maxpool":
            x = tf.keras.layers.MaxPooling2D((2, 2), name=f"{name_prefix}_pool{i+1}")(x)
        elif pool_type == "avgpool":
            x = tf.keras.layers.AveragePooling2D((2, 2), name=f"{name_prefix}_pool{i+1}")(x)

    gp = cfg.get("global_pooling", "global_average")
    if gp == "global_average":
        x = tf.keras.layers.GlobalAveragePooling2D(name=f"{name_prefix}_gap")(x)
    elif gp == "global_max":
        x = tf.keras.layers.GlobalMaxPooling2D(name=f"{name_prefix}_gmp")(x)
    elif gp == "flatten":
        x = tf.keras.layers.Flatten(name=f"{name_prefix}_flatten")(x)

    for i, dense in enumerate(cfg.get("dense_layers", [])):
        x = tf.keras.layers.Dense(dense["units"], activation=dense.get("activation", "relu"), name=f"{name_prefix}_dense{i+1}")(x)
        if "dropout" in dense:
            x = tf.keras.layers.Dropout(dense["dropout"], name=f"{name_prefix}_dropout{i+1}")(x)

    output = tf.keras.layers.Dense(cfg["output_units"], activation=cfg.get("output_activation", "sigmoid"), name=f"{name_prefix}_out")(x)

    return tf.keras.models.Model(inputs=inputs, outputs=output)
