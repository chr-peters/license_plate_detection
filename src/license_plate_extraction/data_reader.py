from license_plate_extraction import settings
from license_plate_extraction.visualization_tools import show_image
import numpy as np
from bs4 import BeautifulSoup
from pathlib import Path
import tensorflow as tf
from typing import List


def get_bounding_box_xml_path_from_image_path(path: Path) -> Path:
    xml_path = path.with_suffix(".xml")
    return xml_path


def get_bounding_box_from_xml_path(path: Path) -> np.ndarray:
    """
    Returns np.array([x_min, y_min, x_max, y_max])
    """
    with open(path, mode="r") as file:
        bs = BeautifulSoup(file, "xml")

    x_min = float(bs.bndbox.xmin.string)
    x_max = float(bs.bndbox.xmax.string)
    y_min = float(bs.bndbox.ymin.string)
    y_max = float(bs.bndbox.ymax.string)

    return np.array([x_min, y_min, x_max, y_max])


def read_image_as_tensor(path: Path) -> tf.Tensor:
    """
    Returns a 3D Tensor object representing the image as RGB.

    Details here:
    https://www.tensorflow.org/tutorials/load_data/images#using_tfdata_for_finer_control
    """
    img_data = tf.io.read_file(str(path))
    img_tensor = tf.io.decode_jpeg(img_data, channels=3)

    return img_tensor


def get_image_paths_from_directory(directory_path: Path) -> List[Path]:
    image_paths = list(directory_path.glob("*_car_*.jpg"))

    return image_paths


if __name__ == "__main__":
    sample_path = settings.DATA_DIR / "eu_cars+lps" / "1T43213_car_eu.jpg"
    bounding_box_path = get_bounding_box_xml_path_from_image_path(sample_path)

    img_tensor = read_image_as_tensor(sample_path)

    bounding_box = get_bounding_box_from_xml_path(bounding_box_path)

    show_image(img_tensor, bounding_box)

    # all_eu_image_paths = get_image_paths_from_directory(
    #     settings.DATA_DIR / "eu_cars+lps"
    # )
    # for cur_path in all_eu_image_paths:
    #     cur_img_tensor = read_image_as_tensor(cur_path)
    #     cur_bounding_box_path = get_bounding_box_xml_path_from_image_path(cur_path)
    #     cur_bounding_box = get_bounding_box_from_xml_path(cur_bounding_box_path)

    #     show_image(cur_img_tensor, cur_bounding_box)