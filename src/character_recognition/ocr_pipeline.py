from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import pytesseract
import ocr_functions

###############################################################################


def ocr_pipeline(img, bounding_box):
    # Setze cmd auf das Verzeichnis, in dem auch Tesseract drin ist
    TESSERACT_DIR = Path(__file__).parent / "bin"
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_DIR / "tesseract.exe"

    methods = [
        "normal",
        "up",
        "down",
        "left",
        "right",
        "topright",
        "bottomright",
        "bottomleft",
        "topleft",
    ]

    confi_frame = pd.DataFrame()

    for m in methods:
        data_plate = ocr_functions.ocr(img, bounding_box, m)
        confi_frame = pd.concat([confi_frame, data_plate])

    char = ocr_functions.ocr_validation(confi_frame)

    return char


###############################################################################

if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent.parent / "data"
    char = ocr_pipeline(
        cv2.imread(str(data_dir / "validation_eu" / "RK340AO_car_eu.jpg")),
        predicted_bounding_box,
    )
    print(char)

    # from license_plate_extraction.prediction import predict_bounding_box_using_mask
    # from license_plate_extraction import data_reader
    # from license_plate_extraction import settings
    # from license_plate_extraction import preprocessing
    # from license_plate_extraction.visualization_tools import show_image

    # image_dir_vali_eu = settings.DATA_DIR / "validation_eu"
    # image_dir_vali_ro = settings.DATA_DIR / "validation_ro"

    # # image_paths = data_reader.get_image_paths_from_directory(
    # #     image_dir_vali_ro, contains="_car_"
    # # )
    # image_paths = data_reader.get_image_paths_from_directory(
    #     image_dir_vali_eu, contains="RK340AO_car_eu"
    # )

    # for cur_path in image_paths:
    #     image_tensor = data_reader.read_image_as_tensor(cur_path)
    #     predicted_bounding_box = predict_bounding_box_using_mask(image_tensor)
    #     image_numpy = image_tensor.numpy()
    #     cur_prediction = ocr_pipeline(image_numpy, predicted_bounding_box)
    #     print(cur_prediction)

    #     cur_image_tensor = data_reader.read_image_as_tensor(cur_path)
    #     visualization_tools.show_image(img=cur_image_tensor, plate_text=cur_prediction)
