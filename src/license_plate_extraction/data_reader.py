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


def resize_bounding_box_with_pad(
    bounding_box, img_width, img_height, target_width, target_height
) -> np.ndarray:
    x_min, y_min, x_max, y_max = bounding_box

    img_aspect_ratio = img_width / img_height
    target_aspect_ratio = target_width / target_height

    if target_aspect_ratio >= img_aspect_ratio:
        # the target image is wider which means that the sides are padded
        y_min_new = y_min / img_height * target_height
        y_max_new = y_max / img_height * target_height

        img_width_new = img_aspect_ratio * target_height
        left_padding = (target_width - img_width_new) / 2

        x_min_new = left_padding + x_min / img_width * img_width_new
        x_max_new = left_padding + x_max / img_width * img_width_new
    else:
        # the target image is taller which means that top and bottom is padded
        x_min_new = x_min / img_width * target_width
        x_max_new = x_max / img_width * target_width

        img_height_new = 1 / img_aspect_ratio * target_width
        top_padding = (target_height - img_height_new) / 2

        y_min_new = top_padding + y_min / img_height * img_height_new
        y_max_new = top_padding + y_max / img_height * img_height_new

    return np.array([x_min_new, y_min_new, x_max_new, y_max_new])


def make_dataset_from_image_paths(
    image_path_list: List[Path], target_img_width, target_img_height
) -> tf.data.Dataset:
    # read all the images as tensors
    image_tensors_list = [
        read_image_as_tensor(cur_img_path) for cur_img_path in image_path_list
    ]

    # resize them
    image_tensors_list_resized = [
        tf.image.resize_with_pad(
            cur_image_tensor,
            target_height=target_img_height,
            target_width=target_img_width,
            antialias=True,
        )
        for cur_image_tensor in image_tensors_list
    ]

    # read all the bounding boxes
    bounding_boxes_list = [
        get_bounding_box_from_xml_path(
            get_bounding_box_xml_path_from_image_path(cur_img_path)
        )
        for cur_img_path in image_path_list
    ]

    bounding_boxes_list_resized = [
        resize_bounding_box_with_pad(
            cur_bounding_box,
            img_width=image_tensors_list[i].shape[1],
            img_height=image_tensors_list[i].shape[0],
            target_width=target_img_width,
            target_height=target_img_height,
        )
        for i, cur_bounding_box in enumerate(bounding_boxes_list)
    ]

    images = tf.image.rgb_to_grayscale(tf.stack(image_tensors_list_resized))
    bounding_boxes = tf.stack(bounding_boxes_list_resized)

    dataset = tf.data.Dataset.from_tensor_slices((images, bounding_boxes))

    return dataset


if __name__ == "__main__":
    # images_directory = settings.DATA_DIR / "eu_cars+lps"
    images_directory = settings.DATA_DIR / "us_cars+lps"
    image_path_list = get_image_paths_from_directory(images_directory)

    dataset = make_dataset_from_image_paths(
        image_path_list, target_img_width=500, target_img_height=500
    )

    example_list = list(dataset.as_numpy_iterator())

    for cur_example in example_list:
        show_image(cur_example[0].astype(int), bounding_box=cur_example[1])
