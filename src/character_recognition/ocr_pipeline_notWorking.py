import cv2
import numpy as np
import pandas as pd
import pytesseract
from pathlib import Path
import ocr_functions_notWorking as ocr_functions
import matplotlib as plt

# from license_plate_extraction.prediction import predict_bounding_box
# from license_plate_extraction import data_reader
# from license_plate_extraction import settings
# from license_plate_extraction import preprocessing
# from license_plate_extraction import visualization_tools

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
    plate_nums = []
    confi_frame = pd.DataFrame()

    for m in methods:
        data_plate, plate_num = ocr_functions.ocr(img, bounding_box, m)
        if plate_num is not None:
            plate_nums.append(plate_num)
        if not data_plate.empty:
            confi_frame = pd.concat([confi_frame, data_plate])

    # char = ocr_functions.ocr_validation(confi_frame, plate_nums, methods)

    return plate_nums, confi_frame


if __name__ == "__main__":
    # a = ocr_pipeline(
    #         cv2.imread(
    #         "G:\Statistik\FallstudienII\Projekt2\Code\Daten\eu_cars+lps/BS47040_car_eu.jpg"
    #         ),
    #         (92 / 600, 201 / 387, (229 - 201) / 387, (214 - 92) / 600),
    #         )
    # print(a)
    # a = ocr_pipeline(
    #     cv2.imread(
    #         "G:\Statistik\FallstudienII\Projekt2\Code\Daten\eu_cars+lps/W053011_car_eu.jpg"
    #     ),
    #     (391 / 600, 202 / 425, (522 - 391) / 600, (232 - 202) / 425),
    #     # modifiziert (nach oben verschoben):
    #     #(371 / 600, 192 / 425, (522 - 371) / 600, (242 - 192) / 425)
    # )
    # print(a)
    data_dir = Path(__file__).parent.parent.parent / "data"
    a = ocr_pipeline(
        cv2.imread(str(data_dir / "eu_cars+lps" / "BIMMIAN_car_eu.jpg")),
        (104 / 711, 210 / 450, (609 - 104) / 711, (326 - 210) / 450),
    )
    print(a)

# if __name__ == "__main__":
#     image_dir_eu = settings.DATA_DIR / "eu_cars+lps"
#     image_dir_no_labels = settings.DATA_DIR / "no_labels"

#     # image_paths = data_reader.get_image_paths_from_directory(
#     #     image_dir_eu, contains="_car_"
#     # )
#     image_paths = data_reader.get_image_paths_from_directory(image_dir_eu)

#     # test_path = image_dir_eu / "BS47040_car_eu.jpg"
#     # make_prediction(test_path)

#     for cur_path in image_paths:
#         image_tensor = data_reader.read_image_as_tensor(cur_path)
#         predicted_bounding_box = predict_bounding_box(image_tensor)
#         image_numpy = image_tensor.numpy()
#         cur_prediction = ocr_pipeline(image_numpy, predicted_bounding_box)
#         print(cur_prediction)

#         cur_image_tensor = data_reader.read_image_as_tensor(cur_path)
#         visualization_tools.show_image(
#             img = cur_image_tensor, plate_text = cur_prediction
#         )
