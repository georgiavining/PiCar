import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import os
import numpy as np

def build_angle_model(
    input_shape=(224, 224, 3),
    conv_filters=[64, 128, 256, 512],
    dense_units=[256, 128],
    dropout=0.3
):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs

    for i, filters in enumerate(conv_filters):
        x = tf.keras.layers.Conv2D(filters, (3,3), padding='same', name=f'conv{i+1}')(x)
        x = tf.keras.layers.BatchNormalization(name=f'bn{i+1}')(x)
        x = tf.keras.layers.Activation('relu', name=f'act{i+1}')(x)
        x = tf.keras.layers.MaxPooling2D((2,2), name=f'pool{i+1}')(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    for i, units in enumerate(dense_units):
        x = tf.keras.layers.Dense(units, activation='relu', name=f'dense{i+1}')(x)
        x = tf.keras.layers.Dropout(dropout, name=f'dropout{i+1}')(x)

    # Linear output — angle is continuous regression
    angle_out = tf.keras.layers.Dense(1, activation='sigmoid', name='angle')(x)

    return tf.keras.models.Model(inputs=inputs, outputs=angle_out)

def build_speed_model(
    input_shape=(224, 224, 3),
    conv_filters=[32, 64, 128, 256],  # lighter — speed is simpler
    dense_units=[128],
    dropout=0.5  # higher dropout — binary task, prone to overfitting
):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs

    for i, filters in enumerate(conv_filters):
        x = tf.keras.layers.Conv2D(filters, (3,3), padding='same', name=f'conv{i+1}')(x)
        x = tf.keras.layers.BatchNormalization(name=f'bn{i+1}')(x)
        x = tf.keras.layers.Activation('relu', name=f'act{i+1}')(x)
        x = tf.keras.layers.MaxPooling2D((2,2), name=f'pool{i+1}')(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    for i, units in enumerate(dense_units):
        x = tf.keras.layers.Dense(units, activation='relu', name=f'dense{i+1}')(x)
        x = tf.keras.layers.Dropout(dropout, name=f'dropout{i+1}')(x)

    # Sigmoid output — speed is binary
    speed_out = tf.keras.layers.Dense(1, activation='sigmoid', name='speed')(x)

    return tf.keras.models.Model(inputs=inputs, outputs=speed_out)

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



    model = tf.keras.Model(inputs=inputs, outputs=[angle_out, speed_out])
    return model
