import numpy as np
from visualization_tools import show_image
import cv2


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


def bounding_box_to_binary_mask(bounding_box_pixel: np.ndarray, img_height, img_width):
    """
    Converts a bounding box to a binary mask of shape [img_height, img_width],
    where 1s encode inside box and 0s encode outside box.
    """
    x_min, y_min, width, height = bounding_box_pixel.astype(int)

    mask = np.zeros(shape=(img_height, img_width))
    mask[y_min : y_min + height, x_min : x_min + width] = 1

    return mask


def mask_to_bounding_box(mask: np.ndarray, threshold_percent=0.1) -> np.ndarray:
    ret, thresh = cv2.threshold(
        (mask * 255).astype("uint8"), int(255 * threshold_percent), 255, 0
    )

    show_image(thresh)

    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    # find biggest rectangle
    bounding_box_max = np.array([0, 0, 0, 0])
    max_size = 0
    for cnt in sorted_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cur_size = w * h
        if cur_size > max_size:
            bounding_box_max = np.array([x, y, w, h])
            max_size = cur_size

    return bounding_box_max


if __name__ == "__main__":
    img_height = 400
    img_width = 600

    test_bounding_box = np.array([100, 150, 300, 200])
    mask = bounding_box_to_binary_mask(test_bounding_box, img_height, img_width)

    show_image(mask)