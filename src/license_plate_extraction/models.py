import tensorflow as tf
from tensorflow.keras import Sequential, Input, layers
from kerastuner.tuners import RandomSearch


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
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(256, 3, padding="same", activation="relu"),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D((2,2)),
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(4, activation="sigmoid", use_bias=False),
        ]
    )

    return model


def test_model_tuner(input_height, input_width, num_channels):
  model = keras.Sequential()
  
  model.add(Input(shape=(input_height, input_width, num_channels)),
            layers.experimental.preprocessing.Rescaling(
                1.0 / 255,
            ))

  model.add(layers.AveragePooling2D())

  for i in range(hp.Int("Conv Layers", min_value=1, max_value=3)):
    model.add(layers.Conv2D(hp.Choice(f"layer_{i}_filters", [16,32,64]), 3, activation="relu"))

  model.add(layers.MaxPool2D(2, 2))
  model.add(layers.Dropout(0.5))
  model.add(layers.Flatten())

  model.add(layers.Dense(hp.Choice("Dense layer", [64, 128, 256, 512, 1025]), activation="relu"))

  model.add(layers.Dense(3, activation="softmax"))

  model.compile(optimizer="adam",
              loss=losses.MSE)
  
  return model


if __name__ == "__main__":
    model = test_model(448, 448, 3)

    print(model.summary())