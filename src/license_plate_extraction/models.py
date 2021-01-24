import tensorflow as tf
from tensorflow.keras import Sequential, Input, layers


def simple_model(input_height, input_width, num_channels):
    model = Sequential(
        [
            Input(shape=(input_height, input_width, num_channels)),
            layers.experimental.preprocessing.Rescaling(
                1.0 / 255,
            ),
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(4, activation="sigmoid"),
        ]
    )

    return model


def tiny_yolo_inspired(input_height, input_width, num_channels):
    model = Sequential(
        [
            Input(shape=(input_height, input_width, num_channels)),
            layers.experimental.preprocessing.Rescaling(
                1.0 / 255,
            ),
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(256, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 1, padding="same", activation="relu"),
            layers.Conv2D(64, 1, padding="same", activation="relu"),
            layers.Conv2D(32, 1, padding="same", activation="relu"),
            layers.Flatten(),
            layers.Dropout(0.1),
            layers.Dense(4, activation="sigmoid"),
        ]
    )

    return model


if __name__ == "__main__":
    model = tiny_yolo_inspired(448, 448, 3)

    print(model.summary())