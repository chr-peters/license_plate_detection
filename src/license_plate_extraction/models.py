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


def tiny_xception(input_height, input_width, num_channels):
    xception = tf.keras.applications.Xception(
        include_top=False,
        weights="imagenet",
        input_shape=(input_height, input_width, num_channels),
    )
    xception.trainable = False

    inputs = tf.keras.Input(shape=(input_height, input_width, num_channels))
    x = tf.keras.applications.xception.preprocess_input(inputs)
    x = xception(x)

    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Conv2D(256, 2, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(64, 2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(32, 2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    outputs = tf.keras.layers.Dense(4, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs)


def simple_mask_predictor(input_height, input_width, num_channels):
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(input_height, input_width, num_channels)),
            tf.keras.layers.experimental.preprocessing.Rescaling(
                1.0 / 255,
            ),
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPool2D(4),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPool2D(4),
            tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid"),
        ]
    )

    return model


if __name__ == "__main__":
    # model = test_model(448, 448, 3)
    # model = efficientnetb2_pretrained(260, 260, 3)
    model = simple_mask_predictor(400, 400, 3)

    model.summary()