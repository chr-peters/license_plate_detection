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


def test_model(input_height, input_width, num_channels):
    model = Sequential(
        [
            Input(shape=(input_height, input_width, num_channels)),
            layers.experimental.preprocessing.Rescaling(
                1.0 / 255,
            ),
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(256, 3, padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(4, activation="sigmoid", use_bias=False),
        ]
    )

    return model


def test_model2(input_height, input_width, num_channels):
    model = Sequential(
        [
            Input(shape=(input_height, input_width, num_channels)),
            layers.experimental.preprocessing.Rescaling(
                1.0 / 255,
            ),
            layers.AveragePooling2D(6, 3),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(4, activation="sigmoid", use_bias=False),
        ]
    )

    return model


def efficientnetb2_pretrained(input_height, input_width, num_channels):
    base_model = tf.keras.applications.EfficientNetB2(
        include_top=False,
        weights="imagenet",
        input_shape=(input_height, input_width, num_channels),
    )
    base_model.trainable = False

    inputs = Input(shape=(input_height, input_width, num_channels))
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base_model(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(4, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)

    return model


if __name__ == "__main__":
    # model = test_model(448, 448, 3)
    model = efficientnetb2_pretrained(260, 260, 3)

    model.summary()