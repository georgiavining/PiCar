from tensorflow.keras import layers, models
import pandas as pd
import os

def build_model1(
    input_shape=(224, 224, 3),
    conv_filters=[32, 64, 128],
    dense_units=128,
    activation='relu',
    final_activation='sigmoid'
):
    inputs = layers.Input(shape=input_shape)
    x = inputs

    # Convolution + Pooling layers
    for i, filters in enumerate(conv_filters):
        x = layers.Conv2D(filters, (3,3), activation=activation, padding='same', name=f'conv{i+1}')(x)
        x = layers.MaxPooling2D((2,2), name=f'pool{i+1}')(x)

    # Flatten and Dense
    x = layers.Flatten()(x)
    x = layers.Dense(dense_units, activation=activation, name='dense1')(x)

    # Multi-output
    angle_out = layers.Dense(1, activation=final_activation, name='angle')(x)
    speed_out = layers.Dense(1, activation=final_activation, name='speed')(x)

    model = models.Model(inputs=inputs, outputs=[angle_out, speed_out])
    return model

def generate_test_predictions(
    model,
    test_dataset,
    predictions_path,
    model_name,
):
    results = []
    image_counter = 0

    for x in test_dataset:
        pred_angle, pred_speed = model.predict(x, verbose=0)

        for j in range(len(x)):
            results.append({
                "image_id": image_counter,
                "angle": float(pred_angle[j][0]),
                "speed": float(pred_speed[j][0])
            })
            image_counter += 1

    df_test_preds = pd.DataFrame(results)

    os.makedirs(predictions_path, exist_ok=True)
    csv_output = os.path.join(predictions_path, f"{model_name}.csv")
    df_test_preds.to_csv(csv_output, index=False)

    print(f"Saved to: {csv_output}")

    return df_test_preds