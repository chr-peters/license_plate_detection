import settings
from visualization_tools import show_image
from preprocessing import (
    bounding_box_to_binary_mask,
    scale_bounding_box,
    bounding_box_in_percent,
    bounding_box_in_pixel,
)
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
    Returns np.array([x_min, y_min, width, height])
    """
    with open(path, mode="r") as file:
        bs = BeautifulSoup(file, "xml")

    x_min = float(bs.bndbox.xmin.string)
    x_max = float(bs.bndbox.xmax.string)
    y_min = float(bs.bndbox.ymin.string)
    y_max = float(bs.bndbox.ymax.string)

    return np.array([x_min, y_min, x_max - x_min, y_max - y_min])


def read_image_as_tensor(path: Path) -> tf.Tensor:
    """
    Returns a 3D Tensor object representing the image as RGB.

    Details here:
    https://www.tensorflow.org/tutorials/load_data/images#using_tfdata_for_finer_control
    """
    img_data = tf.io.read_file(str(path))
    img_tensor = tf.io.decode_jpeg(img_data, channels=3)

    return img_tensor


def get_image_paths_from_directory(directory_path: Path, contains="") -> List[Path]:
    if contains is None or contains == "":
        image_paths = list(directory_path.glob("*.jpg"))
    else:
        image_paths = list(directory_path.glob(f"*{contains}*.jpg"))

    return image_paths


def make_dataset_from_image_paths(
    image_path_list: List[Path], target_img_height, target_img_width
) -> tf.data.Dataset:
    # read all the images as tensors
    image_tensors_list = [
        read_image_as_tensor(cur_img_path) for cur_img_path in image_path_list
    ]

    # resize them
    image_tensors_list_resized = [
        tf.image.resize(
            cur_image_tensor,
            size=(target_img_height, target_img_width),
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

    bounding_boxes_list_resized_percent = [
        bounding_box_in_percent(
            scale_bounding_box(
                cur_bounding_box,
                img_height=image_tensors_list[i].shape[0],
                img_width=image_tensors_list[i].shape[1],
                target_img_height=target_img_height,
                target_img_width=target_img_width,
            ),
            img_height=target_img_height,
            img_width=target_img_width,
        )
        for i, cur_bounding_box in enumerate(bounding_boxes_list)
    ]

    # images = tf.image.rgb_to_grayscale(tf.stack(image_tensors_list_resized))
    images = tf.stack(image_tensors_list_resized)
    bounding_boxes = tf.stack(bounding_boxes_list_resized_percent)

    dataset = tf.data.Dataset.from_tensor_slices((images, bounding_boxes))

    return dataset


def make_dataset_from_image_paths_with_masks(
    image_path_list: List[Path], target_img_height, target_img_width
) -> tf.data.Dataset:
    """
    This also creates a dataset, but this time it uses binary masks as the prediction target instead
    of bounding box arrays.
    """
    # read all the images as tensors
    image_tensors_list = [
        read_image_as_tensor(cur_img_path) for cur_img_path in image_path_list
    ]

    # resize them
    image_tensors_list_resized = [
        tf.image.resize(
            cur_image_tensor,
            size=(target_img_height, target_img_width),
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

    # resize them and convert them to binary masks
    bounding_boxes_list = [
        bounding_box_to_binary_mask(
            scale_bounding_box(
                cur_bounding_box,
                img_height=image_tensors_list[i].shape[0],
                img_width=image_tensors_list[i].shape[1],
                target_img_height=target_img_height,
                target_img_width=target_img_width,
            ),
            img_height=target_img_height,
            img_width=target_img_width,
        )
        for i, cur_bounding_box in enumerate(bounding_boxes_list)
    ]

    # images = tf.image.rgb_to_grayscale(tf.stack(image_tensors_list_resized))
    images = tf.stack(image_tensors_list_resized)
    bounding_boxes = tf.stack(bounding_boxes_list)

    dataset = tf.data.Dataset.from_tensor_slices((images, bounding_boxes))

    return dataset


if __name__ == "__main__":
    # images_directory = settings.DATA_DIR / "cars_russia"
    images_directory = settings.DATA_DIR / "eu_cars+lps"
    image_path_list = get_image_paths_from_directory(images_directory, contains="_car_")

    TARGET_IMG_HEIGHT = 500
    TARGET_IMG_WIDTH = 500

    dataset = make_dataset_from_image_paths_with_masks(
        image_path_list,
        target_img_height=TARGET_IMG_HEIGHT,
        target_img_width=TARGET_IMG_WIDTH,
    )

    example_list = list(dataset.as_numpy_iterator())

    # for cur_example in example_list:
    #     print(cur_example[1])
    #     print(
    #         bounding_box_in_pixel(
    #             cur_example[1],
    #             img_height=TARGET_IMG_HEIGHT,
    #             img_width=TARGET_IMG_WIDTH,
    #         )
    #     )
    #     show_image(
    #         cur_example[0].astype(int),
    #         bounding_box=bounding_box_in_pixel(
    #             cur_example[1],
    #             img_height=TARGET_IMG_HEIGHT,
    #             img_width=TARGET_IMG_WIDTH,
    #         ),
    #     )

    for cur_example in example_list:
        cur_img = cur_example[0]
        cur_mask = cur_example[1]

        show_image(np.multiply(cur_img, cur_mask[:, :, np.newaxis]).astype(int))
