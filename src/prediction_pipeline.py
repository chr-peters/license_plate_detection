from license_plate_extraction.prediction import predict_bounding_box
from license_plate_extraction import data_reader
from license_plate_extraction import settings
from license_plate_extraction import preprocessing
from license_plate_extraction import visualization_tools
from character_recognition.ocr_pipeline import ocr_pipeline
from pathlib import Path


def make_prediction(image_path: Path) -> str:
    """
    :returns: bounding_box (percent), license_plate_string
    """
    image_tensor = data_reader.read_image_as_tensor(image_path)

    predicted_bounding_box = predict_bounding_box(image_tensor)

    image_numpy = image_tensor.numpy()
    ocr_prediction = ocr_pipeline(image_numpy, predicted_bounding_box)

    return predicted_bounding_box, ocr_prediction


if __name__ == "__main__":
    image_dir_eu = settings.DATA_DIR / "eu_cars+lps"

    image_paths = data_reader.get_image_paths_from_directory(image_dir_eu)

    # test_path = image_dir_eu / "BS47040_car_eu.jpg"
    # make_prediction(test_path)

    for cur_path in image_paths:
        cur_bounding_box, cur_prediction = make_prediction(cur_path)
        print(cur_prediction)

        cur_image_tensor = data_reader.read_image_as_tensor(cur_path)
        cur_bounding_box_pixel = preprocessing.bounding_box_in_pixel(
            cur_bounding_box, cur_image_tensor.shape[0], cur_image_tensor.shape[1]
        )
        visualization_tools.show_image(cur_image_tensor, cur_bounding_box_pixel)
