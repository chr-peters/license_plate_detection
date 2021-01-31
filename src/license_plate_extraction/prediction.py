from pathlib import Path
import numpy as np
from data_reader import (
    get_bounding_box_from_xml_path,
    get_bounding_box_xml_path_from_image_path,
)
import settings


def predict_bounding_box(path: Path) -> np.ndarray:
    """
    Predicts a bounding box for an image.

    :param path: The path where the image is located.
    :returns: An np.ndarray containing the bounding box prediction in the format [x_min, y_min, width, height] (Pixel values).
    """
    bounding_box_path = get_bounding_box_xml_path_from_image_path(path)
    bounding_box = get_bounding_box_from_xml_path(bounding_box_path)

    return bounding_box


if __name__ == "__main__":
    example_image_path = settings.DATA_DIR / "eu_cars+lps/1T43213_car_eu.jpg"

    prediction = predict_bounding_box(example_image_path)

    print(prediction)