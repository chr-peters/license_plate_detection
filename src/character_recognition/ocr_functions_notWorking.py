import cv2
import numpy as np
import pytesseract

# from pathlib import Path
import pandas as pd


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


def image_preproc(img_bound):
    gray = cv2.cvtColor(img_bound, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    blur_g = cv2.GaussianBlur(gray, (5, 5), 0)
    blur = cv2.medianBlur(blur_g, 3)
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilation = cv2.dilate(thresh, rect_kern, iterations=1)
    opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, rect_kern)

    contours, hierarchy = cv2.findContours(
        opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    return sorted_contours, gray, thresh


def ocr_extraction(sorted_contours, gray, thresh, x_start, y_start):
    data_plate = pd.DataFrame()
    plate_num = ""
    i = 1

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
        roi = thresh[np.max([y - 5, 0]) : y + h + 5, x - 5 : x + w + 5]
        if roi.size == 0:
            roi = thresh[y - 2 : y + h + 2, x - 2 : x + w + 2]
        if roi.size == 0:
            roi = thresh[y : y + h, x : x + w]
        roi = cv2.bitwise_not(roi)
        roi = cv2.medianBlur(roi, 5)
        dat = pytesseract.image_to_data(
            roi,
            config="-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3",
            output_type="data.frame",
        )
        dat["number"] = i
        dat["x"] = (x / 3) + x_start
        dat["y"] = (y / 3) + y_start
        dat["w"] = w
        dat["h"] = h
        i += 1
        text = pytesseract.image_to_string(
            roi,
            config="-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3",
        )
        data_plate = pd.concat([data_plate, dat])
        plate_num += text

    data_plate = data_plate.dropna()
    if not data_plate.empty:
        data_plate = data_plate.set_index("number")
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
    plate_num = plate_num.splitlines()
    plate_num = "".join(map(str, plate_num))

    return data_plate, plate_num


def ocr(img, bounding_box, method):
    # check which method is used:
    if method == "normal":
        img_bound, x_start, y_start = extract_plate(img, bounding_box)
        sorted_contours, gray, thresh = image_preproc(img_bound)
        data_plate, plate_num = ocr_extraction(
            sorted_contours, gray, thresh, x_start, y_start
        )

    if method == "up":
        img_bound_up, x_startup, y_startup = extract_plate_ver(
            img, bounding_box, int(bounding_box[1] * img.shape[0] * 0.1), True
        )
        sorted_contours, gray, thresh = image_preproc(img_bound_up)
        data_plate, plate_num = ocr_extraction(
            sorted_contours, gray, thresh, x_startup, y_startup
        )

    if method == "down":
        img_bound_down, x_startdown, y_startdown = extract_plate_ver(
            img, bounding_box, int(bounding_box[1] * img.shape[0] * 0.1), False
        )
        sorted_contours, gray, thresh = image_preproc(img_bound)
        data_plate, plate_num = ocr_extraction(
            sorted_contours, gray, thresh, x_start, y_start
        )

    if method == "left":
        img_bound_left, x_startleft, y_startleft = extract_plate_hor(
            img, bounding_box, int(bounding_box[0] * img.shape[1] * 0.1), True
        )
        sorted_contours, gray, thresh = image_preproc(img_bound)
        data_plate, plate_num = ocr_extraction(
            sorted_contours, gray, thresh, x_start, y_start
        )

    if method == "right":
        img_bound_right, x_startright, y_startright = extract_plate_hor(
            img, bounding_box, int(bounding_box[0] * img.shape[1] * 0.1), False
        )
        sorted_contours, gray, thresh = image_preproc(img_bound)
        data_plate, plate_num = ocr_extraction(
            sorted_contours, gray, thresh, x_start, y_start
        )

    if method == "topright":
        img_bound_tr, x_starttr, y_starttr = extract_plate_side(
            img,
            bounding_box,
            int(bounding_box[0] * img.shape[1] * 0.1),
            int(bounding_box[1] * img.shape[0] * 0.1),
            1,
        )
        sorted_contours, gray, thresh = image_preproc(img_bound)
        data_plate, plate_num = ocr_extraction(
            sorted_contours, gray, thresh, x_start, y_start
        )

    if method == "bottomright":
        img_bound_br, x_startbr, y_startbr = extract_plate_side(
            img,
            bounding_box,
            int(bounding_box[0] * img.shape[1] * 0.1),
            int(bounding_box[1] * img.shape[0] * 0.1),
            2,
        )
        sorted_contours, gray, thresh = image_preproc(img_bound)
        data_plate, plate_num = ocr_extraction(
            sorted_contours, gray, thresh, x_start, y_start
        )

    if method == "bottomleft":
        img_bound_bl, x_startbl, y_startbl = extract_plate_side(
            img,
            bounding_box,
            int(bounding_box[0] * img.shape[1] * 0.1),
            int(bounding_box[1] * img.shape[0] * 0.1),
            3,
        )
        sorted_contours, gray, thresh = image_preproc(img_bound)
        data_plate, plate_num = ocr_extraction(
            sorted_contours, gray, thresh, x_start, y_start
        )

    if method == "topleft":
        img_bound_tl, x_starttl, y_starttl = extract_plate_side(
            img,
            bounding_box,
            int(bounding_box[0] * img.shape[1] * 0.1),
            int(bounding_box[1] * img.shape[0] * 0.1),
            4,
        )
        sorted_contours, gray, thresh = image_preproc(img_bound)
        data_plate, plate_num = ocr_extraction(
            sorted_contours, gray, thresh, x_start, y_start
        )

    return data_plate, plate_num


def ocr_validation(data_plate, plate_num, methods):
    candidate = max(plate_num)
    ind = methods[plate_num.index(candidate)]

    longest_frame = data_plate[ind]

    indices = longest_frame.index.tolist()

    for i in indices:
        width = int(longest_frame.width[i])
        height = int(longest_frame.height[i])
        conf = longest_frame.conf[i]
        char = longest_frame.text[i]

        for m in methods:
            cur_frame = data_plate[m]
            if cur_frame.empty:
                continue
            cur_opp = cur_frame[
                (cur_frame.width > width - 2) & (cur_frame.width < width + 2)
            ]
            cur_opp = cur_frame[
                (cur_frame.height > height - 2) & (cur_frame.height < height + 2)
            ]
            if cur_opp.empty:
                continue
            indi = cur_opp.index.tolist()
            if len(cur_opp.index) == 1:
                if (
                    cur_opp.text[indi].to_string()[
                        len(cur_opp.text[indi].to_string()) - 1
                    ]
                    == char
                ):
                    continue
                else:
                    if (cur_opp.conf > conf).bool():
                        candidate = candidate.replace(
                            char,
                            cur_opp.text[indi].to_string()[
                                len(cur_opp.text[indi].to_string()) - 1
                            ],
                        )
                    else:
                        continue
            else:
                for j in indi:
                    if cur_opp.conf[j] > conf:
                        candidate = candidate.replace(
                            char,
                            cur_opp.text[j].to_string()[
                                len(cur_opp.text[j].to_string()) - 1
                            ],
                        )
                    else:
                        continue

    return candidate
