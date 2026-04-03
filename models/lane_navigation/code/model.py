import tensorflow as tf
from tensorflow.keras import layers

def create_efficientnet_model(input_shape=(224, 224, 3)):

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )
    base_model.trainable = True
    
    inputs = layers.Input(shape=input_shape, name="input_layer")
    x = base_model(inputs, training=True)
    x = layers.GlobalAveragePooling2D(name="pooling_layer")(x)
    x       = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x       = layers.Dropout(0.15)(x)
    outputs = layers.Dense(1, activation='sigmoid', name="angle_output")(x)
    model = tf.keras.Model(inputs, outputs)

    return model

def create_mv3_model(input_shape=(224, 224, 3)):
    base_model = tf.keras.applications.MobileNetV3Small(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )
    base_model.trainable = True

    inputs  = layers.Input(shape=input_shape)
    x       = base_model(inputs, training=True)
    x       = layers.Dropout(0.2)(x)
    x       = layers.Dense(128, activation='relu')(x)
    x       = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs, outputs)