import cv2
import numpy as np
import pytesseract
from pathlib import Path

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

    # Aus Bild das Nummernschild extrahieren:
    x = int(round(bounding_box[0] * img.shape[1]))
    y = int(round(bounding_box[1] * img.shape[0]))
    w = int(round(bounding_box[2] * img.shape[1]))
    h = int(round(bounding_box[3] * img.shape[0]))

    img_bound = img[y : y + h, x : x + w]
    # Preprocessing:
    gray = cv2.cvtColor(img_bound, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    blur_g = cv2.GaussianBlur(gray, (5, 5), 0)
    blur = cv2.medianBlur(blur_g, 3)
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilation = cv2.dilate(thresh, rect_kern, iterations=1)
    opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, rect_kern)
    # Konturen:
    contours, hierarchy = cv2.findContours(
        opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    
    img_c = cv2.drawContours(gray, sorted_contours, -1, (0,255,0), 3)

    plate_num = ""
    # gehe ueber Konturen und lese nur solche aus, die Zeichen sind:
    for cnt in sorted_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        height, width = gray.shape        
        if height / float(h) > 4:
            continue
        ratio = h / float(w)
        if ratio < 1.2:
            continue
        if width / float(w) > 50:
            continue  # 25
        rect = cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = thresh[y - 5 : y + h + 5, x - 5 : x + w + 5]
        if roi.size == 0:
            roi = thresh[y - 2 : y + h + 2, x - 2 : x + w + 2]
        if roi.size == 0:
            roi = thresh[y : y + h, x : x + w]
        roi = cv2.bitwise_not(roi)
        roi = cv2.medianBlur(roi, 5)        
        text = pytesseract.image_to_string(
            roi,
            config="-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3",
        )
        plate_num += text

    plate_num = plate_num.splitlines()
    plate_num = "".join(map(str, plate_num))

    return plate_num


if __name__ == "__main__":
    # ocr_pipeline(
    #     cv2.imread(
    #         "G:\Statistik\FallstudienII\Projekt2\Code\Daten\eu_cars+lps/BS47040_car_eu.jpg"
    #     ),
    #     (92 / 600, 201 / 387, (229 - 201) / 387, (214 - 92) / 600),
    # )
    ocr_pipeline(
        cv2.imread(
            "G:\Statistik\FallstudienII\Projekt2\Code\Daten\eu_cars+lps/W053011_car_eu.jpg"
        ),
        (391 / 600, 202 / 425, (522 - 391) / 600, (232 - 202) / 425),
    )

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
        
        
        
        
        
        
        
        