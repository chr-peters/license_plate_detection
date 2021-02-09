import tensorflow as tf
import settings
from data_reader import get_image_paths_from_directory, make_dataset_from_image_paths
from visualization_tools import show_image
from preprocessing import bounding_box_in_pixel
import numpy as np


def flip_bounding_box(bounding_box_percent: np.ndarray) -> np.ndarray:
    x_min, y_min, width, height = bounding_box_percent

    x_min_new = 1 - x_min - width

    return np.array([x_min_new, y_min, width, height])


def contrast_fun(cur_image, cur_bounding_box):
    img_list = [cur_image]
    bounding_box_list = [cur_bounding_box]

    for cur_contrast in [4]:
        cur_contrast_img = tf.image.adjust_contrast(
            cur_image, contrast_factor=cur_contrast
        )

        cur_contrast_img = tf.clip_by_value(
            cur_contrast_img, clip_value_min=0, clip_value_max=255
        )

        img_list.append(cur_contrast_img)
        bounding_box_list.append(cur_bounding_box)

    return (np.array(img_list), np.array(bounding_box_list))


def flip_fun(cur_image, cur_bounding_box):
    img_list = [cur_image, tf.image.flip_left_right(cur_image)]
    bounding_box_list = [cur_bounding_box, flip_bounding_box(cur_bounding_box)]

    return (np.array(img_list), np.array(bounding_box_list))


def brightness_fun(cur_image, cur_bounding_box):
    img_list = [cur_image]
    bounding_box_list = [cur_bounding_box]

    for cur_brightness in [-100, 100]:
        cur_bright_img = tf.image.adjust_brightness(cur_image, delta=cur_brightness)

        cur_bright_img = tf.clip_by_value(
            cur_bright_img, clip_value_min=0, clip_value_max=255
        )

        img_list.append(cur_bright_img)
        bounding_box_list.append(cur_bounding_box)

    return (np.array(img_list), np.array(bounding_box_list))


def _augment_dataset(dataset, augment_fun):
    dataset = dataset.map(
        lambda x, y: tf.py_function(
            augment_fun, inp=[x, y], Tout=(tf.float64, tf.float64)
        )
    )

    dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))

    return dataset


def add_contrast(dataset: tf.data.Dataset) -> tf.data.Dataset:
    return _augment_dataset(dataset, contrast_fun)


def add_brightness(dataset: tf.data.Dataset) -> tf.data.Dataset:
    return _augment_dataset(dataset, brightness_fun)


def horizontal_flip(dataset: tf.data.Dataset):
    return _augment_dataset(dataset, flip_fun)


if __name__ == "__main__":
    images_directory_br = settings.DATA_DIR / "br_cars+lps"
    images_directory_eu = settings.DATA_DIR / "eu_cars+lps"
    images_directory_ro = settings.DATA_DIR / "ro_cars+lps"
    images_directory_us = settings.DATA_DIR / "us_cars+lps"

    image_path_list = [
        # *get_image_paths_from_directory(images_directory_br, contains="_car_"),
        # *get_image_paths_from_directory(images_directory_eu, contains="_car_"),
        # *get_image_paths_from_directory(images_directory_ro, contains="_car_"),
        *get_image_paths_from_directory(images_directory_us, contains="_car_"),
    ]

    TARGET_IMG_HEIGHT = 500
    TARGET_IMG_WIDTH = 500

    dataset = make_dataset_from_image_paths(
        image_path_list,
        target_img_height=TARGET_IMG_HEIGHT,
        target_img_width=TARGET_IMG_WIDTH,
    )

    print(f"Num images original: {len(dataset)}")

    dataset_augmented = dataset
    dataset_augmented = horizontal_flip(dataset_augmented)
    dataset_augmented = add_brightness(dataset_augmented)
    dataset_augmented = add_contrast(dataset_augmented)

    print(f"Num images augmented: {len(list(dataset_augmented.as_numpy_iterator()))}")

    for cur_example in dataset_augmented.as_numpy_iterator():
        cur_image = cur_example[0]
        cur_bounding_box = cur_example[1]

        cur_bounding_box_pixel = bounding_box_in_pixel(
            cur_bounding_box, TARGET_IMG_HEIGHT, TARGET_IMG_WIDTH
        )

        show_image(cur_image.astype(int), cur_bounding_box_pixel)