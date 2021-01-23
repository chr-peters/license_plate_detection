from license_plate_extraction import preprocessing
import numpy as np


def test_bounding_box_percent():
    bounding_box = np.array([213, 214, 314, 237])

    img_height = 432
    img_width = 576

    bounding_box_percent = preprocessing.bounding_box_in_percent(
        bounding_box, img_height, img_width
    )

    assert (
        np.allclose(
            bounding_box_percent, np.array([213 / 576, 214 / 432, 314 / 576, 237 / 432])
        )
        == True
    )


def test_bounding_box_pixel():
    bounding_box = np.array([213 / 576, 214 / 432, 314 / 576, 237 / 432])

    img_height = 432
    img_width = 576

    bounding_box_pixel = preprocessing.bounding_box_in_pixel(
        bounding_box, img_height, img_width
    )

    assert np.allclose(bounding_box_pixel, np.array([213, 214, 314, 237])) == True
