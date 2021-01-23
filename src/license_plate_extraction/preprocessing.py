import numpy as np


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


def bounding_box_in_percent(bounding_box, img_height, img_width):
    x_min, y_min, x_max, y_max = bounding_box

    x_min_new = x_min / img_width
    x_max_new = x_max / img_width
    y_min_new = y_min / img_height
    y_max_new = y_max / img_height

    return np.array([x_min_new, y_min_new, x_max_new, y_max_new])


def bounding_box_in_pixel(bounding_box_in_percent, img_height, img_width):
    x_min, y_min, x_max, y_max = bounding_box_in_percent

    x_min_new = x_min * img_width
    x_max_new = x_max * img_width
    y_min_new = y_min * img_height
    y_max_new = y_max * img_height

    return np.array([x_min_new, y_min_new, x_max_new, y_max_new])


def scale_bounding_box(
    bounding_box, img_height, img_width, target_img_height, target_img_width
):
    bounding_box_percent = bounding_box_in_percent(bounding_box, img_height, img_width)

    x_min, y_min, x_max, y_max = bounding_box_percent

    return np.array(
        [
            x_min * target_img_width,
            y_min * target_img_height,
            x_max * target_img_width,
            y_max * target_img_height,
        ]
    )
