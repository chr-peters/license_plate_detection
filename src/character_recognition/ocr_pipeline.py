from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import pytesseract
import ocr_functions_notWorking as ocr_functions

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
    from license_plate_extraction.prediction import predict_bounding_box
    from license_plate_extraction import data_reader
    from license_plate_extraction import settings
    from license_plate_extraction import preprocessing
    from license_plate_extraction import visualization_tools

    image_dir_vali_eu = settings.DATA_DIR / "validation_eu"
    image_dir_vali_ro = settings.DATA_DIR / "validation_ro"

    #image_paths = data_reader.get_image_paths_from_directory(image_dir_vali_eu)
    image_paths = data_reader.get_image_paths_from_directory(image_dir_vali_ro)

    for cur_path in image_paths:
        image_tensor = data_reader.read_image_as_tensor(cur_path)
        predicted_bounding_box = predict_bounding_box(image_tensor)
        image_numpy = image_tensor.numpy()
        cur_prediction = ocr_pipeline(image_numpy, predicted_bounding_box)
        print(cur_prediction)

        cur_image_tensor = data_reader.read_image_as_tensor(cur_path)
        visualization_tools.show_image(
            img = cur_image_tensor, plate_text = cur_prediction
        )
        
    #data_dir = Path(__file__).parent.parent.parent / "data"
    #a = ocr_pipeline(
    #        cv2.imread(str(data_dir / "eu_cars+lps" / "BIMMIAN_car_eu.jpg")),
    #        (104 / 711, 210 / 450, (609 - 104) / 711, (326 - 210) / 450),
    #     )
    #print(a)