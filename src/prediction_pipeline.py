from license_plate_extraction.prediction import predict_bounding_box
from license_plate_extraction import data_reader
from license_plate_extraction import settings
from license_plate_extraction import preprocessing
from license_plate_extraction import visualization_tools
from character_recognition.ocr_pipeline import ocr_pipeline
from pathlib import Path


def make_prediction(image_path: Path) -> str:
    image_tensor = data_reader.read_image_as_tensor(image_path)

    # TODO: predict bounding box.
    # for now: use the true bounding box
    predicted_bounding_box = data_reader.get_bounding_box_from_xml_path(
        data_reader.get_bounding_box_xml_path_from_image_path(image_path)
    )
    img_height = image_tensor.shape[0]
    img_width = image_tensor.shape[1]
    predicted_bounding_box = preprocessing.bounding_box_in_percent(
        predicted_bounding_box, img_height, img_width
    )

    image_numpy = image_tensor.numpy()

    prediction = ocr_pipeline(image_numpy, predicted_bounding_box)

    return prediction


if __name__ == "__main__":
    image_dir_eu = settings.DATA_DIR / "eu_cars+lps"

    image_paths = data_reader.get_image_paths_from_directory(image_dir_eu)

    #test_path = image_dir_eu / "BS47040_car_eu.jpg"
    #make_prediction(test_path)

    for cur_path in image_paths:
        cur_prediction = make_prediction(cur_path)
        print(cur_prediction)

        cur_image_tensor = data_reader.read_image_as_tensor(cur_path)
        visualization_tools.show_image(cur_image_tensor)
