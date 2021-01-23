import numpy as np


def bounding_box_in_percent(bounding_box, img_height, img_width):
    x_min, y_min, width, height = bounding_box

    x_min_new = x_min / img_width
    y_min_new = y_min / img_height
    width_new = width / img_width
    height_new = height / img_height

    return np.array([x_min_new, y_min_new, width_new, height_new])


def bounding_box_in_pixel(bounding_box_in_percent, img_height, img_width):
    x_min, y_min, width, height = bounding_box_in_percent

    x_min_new = x_min * img_width
    y_min_new = y_min * img_height
    width_new = width * img_width
    height_new = height * img_height

    return np.array([x_min_new, y_min_new, width_new, height_new])


def scale_bounding_box(
    bounding_box, img_height, img_width, target_img_height, target_img_width
):
    bounding_box_percent = bounding_box_in_percent(bounding_box, img_height, img_width)

    x_min, y_min, width, height = bounding_box_percent

    return np.array(
        [
            x_min * target_img_width,
            y_min * target_img_height,
            width * target_img_width,
            height * target_img_height,
        ]
    )
