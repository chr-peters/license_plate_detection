from license_plate_extraction import data_reader, settings


def test_bounding_boxes():
    """
    This tests if all the bounding boxes are within the dimensions of an image.
    """

    img_path_list = [
        *data_reader.get_image_paths_from_directory(
            settings.DATA_DIR / "br_cars+lps", contains="_car_"
        ),
        *data_reader.get_image_paths_from_directory(
            settings.DATA_DIR / "eu_cars+lps", contains="_car_"
        ),
        *data_reader.get_image_paths_from_directory(
            settings.DATA_DIR / "ro_cars+lps", contains="_car_"
        ),
        *data_reader.get_image_paths_from_directory(
            settings.DATA_DIR / "us_cars+lps", contains="_car_"
        ),
        *data_reader.get_image_paths_from_directory(
            settings.DATA_DIR / "validation_eu", contains="_car_"
        ),
        *data_reader.get_image_paths_from_directory(
            settings.DATA_DIR / "validation_ro", contains="_car_"
        ),
        *data_reader.get_image_paths_from_directory(settings.DATA_DIR / "cars_russia"),
    ]

    for cur_path in img_path_list:
        cur_image = data_reader.read_image_as_tensor(cur_path)
        cur_bounding_box = data_reader.get_bounding_box_from_xml_path(
            data_reader.get_bounding_box_xml_path_from_image_path(cur_path)
        )

        x_min, y_min, width, height = cur_bounding_box

        assert x_min >= 0, cur_path
        assert x_min <= cur_image.shape[1], cur_path
        assert y_min >= 0, cur_path
        assert y_min <= cur_image.shape[0], cur_path
        assert width >= 0, cur_path
        assert width <= cur_image.shape[1] - x_min, cur_path
        assert height >= 0, cur_path
        assert height <= cur_image.shape[0] - y_min, cur_path