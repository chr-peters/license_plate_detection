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
        cv2.imread(str(data_dir / "validation_eu" / "LM025BD_car_eu.jpg")),
        (140.9375 / 451, 224.77 / 364, 115.005 / 451, 38.22 / 364),
    )
    print(char)
