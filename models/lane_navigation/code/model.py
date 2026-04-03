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
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(1, name="angle_output")(x)
    model = tf.keras.Model(inputs, outputs)

    return model