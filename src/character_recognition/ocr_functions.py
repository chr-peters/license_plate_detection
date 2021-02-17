import cv2
import numpy as np
import pytesseract
import pandas as pd

###############################################################################


def extract_plate(img, bounding_box):
    x = int(round(bounding_box[0] * img.shape[1]))
    y = int(round(bounding_box[1] * img.shape[0]))
    w = int(round(bounding_box[2] * img.shape[1]))
    h = int(round(bounding_box[3] * img.shape[0]))

    img_bound = img[y : y + h, x : x + w]
    x_start = x
    y_start = y

    return img_bound, x_start, y_start


def extract_plate_hor(img, bounding_box, change, left):
    x = int(round(bounding_box[0] * img.shape[1]))
    y = int(round(bounding_box[1] * img.shape[0]))
    w = int(round(bounding_box[2] * img.shape[1]))
    h = int(round(bounding_box[3] * img.shape[0]))

    if left:
        img_bound = img[y : y + h, x - change : x + w - change]
        x_start = x - change
        y_start = y
    else:
        img_bound = img[y : y + h, x + change : x + w + change]
        x_start = x + change
        y_start = y

    return img_bound, x_start, y_start


def extract_plate_ver(img, bounding_box, change, up):
    x = int(round(bounding_box[0] * img.shape[1]))
    y = int(round(bounding_box[1] * img.shape[0]))
    w = int(round(bounding_box[2] * img.shape[1]))
    h = int(round(bounding_box[3] * img.shape[0]))

    if up:
        img_bound = img[y - change : y + h - change, x : x + w]
        x_start = x
        y_start = y - change
    else:
        img_bound = img[y + change : y + h + change, x : x + w]
        x_start = x
        y_start = y + change

    return img_bound, x_start, y_start


def extract_plate_side(img, bounding_box, change_x, change_y, coords):
    x = int(round(bounding_box[0] * img.shape[1]))
    y = int(round(bounding_box[1] * img.shape[0]))
    w = int(round(bounding_box[2] * img.shape[1]))
    h = int(round(bounding_box[3] * img.shape[0]))

    if coords == 1:  # topright
        img_bound = img[
            y - change_y : y + h - change_y, x + change_x : x + w + change_x
        ]
        x_start = x + change_x
        y_start = y - change_y
    if coords == 2:  # bottomright
        img_bound = img[
            y + change_y : y + h + change_y, x + change_x : x + w + change_x
        ]
        x_start = x + change_x
        y_start = y + change_y
    if coords == 3:  # bottomleft
        img_bound = img[
            y + change_y : y + h + change_y, x - change_x : x + w - change_x
        ]
        x_start = x - change_x
        y_start = y + change_y
    if coords == 4:  # topleft
        img_bound = img[
            y - change_y : y + h - change_y, x - change_x : x + w - change_x
        ]
        x_start = x - change_x
        y_start = y - change_y

    return img_bound, x_start, y_start


def ocr_extraction(img_bound, x_start, y_start):
    method = ["otsu", "60", "80", "100", "120"]

    data_plate = pd.DataFrame()

    for m in method:
        gray = cv2.cvtColor(img_bound, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        blur_g = cv2.GaussianBlur(gray, (5, 5), 0)
        blur = cv2.medianBlur(blur_g, 3)
        if m == "otsu":
            ret, thresh = cv2.threshold(
                blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
        else:
            ret, thresh = cv2.threshold(blur, int(m), 255, cv2.THRESH_BINARY_INV)
        rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilation = cv2.dilate(thresh, rect_kern, iterations=1)

        contours, hierarchy = cv2.findContours(
            dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

        for cnt in sorted_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            height, width = gray.shape
            if height / float(h) > 3:
                continue
            ratio = h / float(w)
            if ratio < 1.2:
                continue
            if width / float(w) > 50:
                continue
            roi = thresh[np.max([y - 5, 0]) : y + h + 5, np.max([x - 5, 0]) : x + w + 5]
            roi = cv2.bitwise_not(roi)
            roi = cv2.medianBlur(roi, 5)
            dat = pytesseract.image_to_data(
                roi,
                config="-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 10 --oem 3",
                output_type="data.frame",
            )
            dat = dat[dat.conf == np.max(dat.conf)]
            dat["x"] = (x / 3) + x_start
            dat["y"] = (y / 3) + y_start
            dat["w"] = w
            dat["h"] = h
            data_plate = pd.concat([data_plate, dat])

    data_plate = data_plate.dropna()
    if not data_plate.empty:
        data_plate = data_plate.drop(
            columns=[
                "level",
                "page_num",
                "block_num",
                "par_num",
                "line_num",
                "word_num",
                "left",
                "top",
                "width",
                "height",
            ]
        )

    return data_plate


def ocr(img, bounding_box, method):
    if set(bounding_box) <= set((0, 0, 0, 0)):
        bounding_box = (
            (0.5 * img.shape[1]) / img.shape[1],
            (0.5 * img.shape[0]) / img.shape[0],
            (0.2 * img.shape[1]) / img.shape[1],
            (0.05 * img.shape[0]) / img.shape[0],
        )

    if method == "normal":
        img_bound, x_start, y_start = extract_plate(img, bounding_box)
        data_plate = ocr_extraction(img_bound, x_start, y_start)

    if method == "up":
        img_bound, x_start, y_start = extract_plate_ver(
            img, bounding_box, int(bounding_box[1] * img.shape[0] * 0.1), True
        )
        data_plate = ocr_extraction(img_bound, x_start, y_start)

    if method == "down":
        img_bound, x_start, y_start = extract_plate_ver(
            img, bounding_box, int(bounding_box[1] * img.shape[0] * 0.1), False
        )
        data_plate = ocr_extraction(img_bound, x_start, y_start)

    if method == "left":
        img_bound, x_start, y_start = extract_plate_hor(
            img, bounding_box, int(bounding_box[0] * img.shape[1] * 0.1), True
        )
        data_plate = ocr_extraction(img_bound, x_start, y_start)

    if method == "right":
        img_bound, x_start, y_start = extract_plate_hor(
            img, bounding_box, int(bounding_box[0] * img.shape[1] * 0.1), False
        )
        data_plate = ocr_extraction(img_bound, x_start, y_start)

    if method == "topright":
        img_bound, x_start, y_start = extract_plate_side(
            img,
            bounding_box,
            int(bounding_box[0] * img.shape[1] * 0.1),
            int(bounding_box[1] * img.shape[0] * 0.1),
            1,
        )
        data_plate = ocr_extraction(img_bound, x_start, y_start)

    if method == "bottomright":
        img_bound, x_start, y_start = extract_plate_side(
            img,
            bounding_box,
            int(bounding_box[0] * img.shape[1] * 0.1),
            int(bounding_box[1] * img.shape[0] * 0.1),
            2,
        )
        data_plate = ocr_extraction(img_bound, x_start, y_start)

    if method == "bottomleft":
        img_bound, x_start, y_start = extract_plate_side(
            img,
            bounding_box,
            int(bounding_box[0] * img.shape[1] * 0.1),
            int(bounding_box[1] * img.shape[0] * 0.1),
            3,
        )
        data_plate = ocr_extraction(img_bound, x_start, y_start)

    if method == "topleft":
        img_bound, x_start, y_start = extract_plate_side(
            img,
            bounding_box,
            int(bounding_box[0] * img.shape[1] * 0.1),
            int(bounding_box[1] * img.shape[0] * 0.1),
            4,
        )
        data_plate = ocr_extraction(img_bound, x_start, y_start)

    return data_plate


def ocr_validation(data_plate):
    data_plate = data_plate.reset_index()

    final_frame = pd.DataFrame()

    for i in range(len(data_plate)):
        if i not in data_plate.index:
            continue

        curr_frame = data_plate[
            (data_plate["x"] <= data_plate.x[i] + 1)
            & (data_plate["x"] >= data_plate.x[i] - 1)
            & (data_plate["y"] <= data_plate.y[i] + 1)
            & (data_plate["y"] >= data_plate.y[i] - 1)
        ]
        data_plate.drop(curr_frame.index, inplace=True)

        best_conf = curr_frame[curr_frame["conf"] == np.max(curr_frame["conf"])]
        if len(best_conf) > 1:
            best_conf = best_conf.head(1)
        final_frame = pd.concat([final_frame, best_conf])

    if not final_frame.empty:
        final_frame = final_frame[final_frame["conf"] >= 40]
        final_frame = final_frame.sort_values(by=["x"])

        for i in final_frame.index:  # range(len(final_frame)):
            if i not in final_frame.index:
                continue
            curr_frame = final_frame[
                (final_frame["x"] <= final_frame.x[i] + 1)
                & (final_frame["x"] >= final_frame.x[i] - 1)
            ]
            if len(curr_frame) == 1:
                continue
            if len(curr_frame.text.unique()) == 1:
                curr_frame = curr_frame.tail(1)
            else:
                curr_frame = curr_frame[
                    curr_frame["conf"] != np.max(curr_frame["conf"])
                ]
            final_frame.drop(curr_frame.index, inplace=True)

        plate = ""
        for char in final_frame.text:
            if isinstance(char, str):
                plate += char
            else:
                plate += str(int(char))

        return plate
    else:
        return ""


###############################################################################

if __name__ == "__main__":
    from pathlib import Path

    data_dir = Path(__file__).parent.parent.parent / "data"
    # img = cv2.imread(str(data_dir / "validation_eu" / "LM633BD_car_eu.jpg"))
    # bounding_box = (162.5025 / 461, 151.375 / 346, 137.1475 / 461, 40.655 / 346)

    img = cv2.imread(str(data_dir / "validation_eu" / "LM025BD_car_eu.jpg"))
    bounding_box = (140.9375 / 451, 224.77 / 364, 115.005 / 451, 38.22 / 364)

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
        data_plate = ocr(img, bounding_box, m)
        confi_frame = pd.concat([confi_frame, data_plate])

    char = ocr_validation(confi_frame)

    print(char)
