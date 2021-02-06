from pathlib import Path
import numpy as np
import tensorflow as tf
import data_reader
import settings


PREDICTION_MODEL_PATH = settings.MODELS_DIR / "test_01.tf"


def predict_bounding_box(
    image_path: Path, model_path=PREDICTION_MODEL_PATH
) -> np.ndarray:
    """
    Predicts a bounding box for an image.

    :param image_path: The path where the image is located.
    :param model_path: The path where the trained keras model is located.
    :returns: An np.ndarray containing the bounding box prediction in the format [x_min, y_min, width, height] (Pixel values).
    """
    model = tf.keras.models.load_model(model_path)

    image_tensor = data_reader.read_image_as_tensor(image_path)

    # resize image to match model input
    input_height = model.inputs[0].shape[1]
    input_width = model.inputs[0].shape[2]
    image_tensor = tf.image.resize(
        image_tensor, size=(input_height, input_width), antialias=True
    )

    image_tensor = tf.reshape(image_tensor, shape=(1, input_height, input_width, 3))

    prediction = model.predict(image_tensor)

    return prediction[0]


if __name__ == "__main__":
    example_image_path = settings.DATA_DIR / "eu_cars+lps/1T43213_car_eu.jpg"

    prediction = predict_bounding_box(example_image_path)

    print(prediction)