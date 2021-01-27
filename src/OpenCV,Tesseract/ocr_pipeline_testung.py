import cv2
import numpy as np
import pytesseract
from Levenshtein import distance as levenshtein_distance
import pylev
import statistics

# setze cmd auf das Verzeichnis, in dem auch Tesseract drin ist
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# lade ausserdem Dateinamen_Listen.spydata und xml_nummernschilder_einlesen.spydata
# per drag&drop in den Variable explorer!


def ocr_one(img, bounding_box, schwellenwert, character):
    # bounding box + gray scaling
    gray = cv2.resize(img, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
    blur_g = cv2.GaussianBlur(gray, (5,5), 0)
    blur = cv2.medianBlur(blur_g, 3)
    ret, thresh = cv2.threshold(blur, schwellenwert, 255, cv2.THRESH_BINARY_INV)
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilation = cv2.dilate(thresh, rect_kern, iterations = 1)
    opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, rect_kern)
    # contours:
    contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    # contours with letters:
    plate_num = ""
    # loop through contours and find letters in license plate
    for cnt in sorted_contours:
        x,y,w,h = cv2.boundingRect(cnt)
        height, width = gray.shape
            # if height of box is not a quarter of total height then skip
        if height / float(h) > 4: continue #6
        ratio = h / float(w)
        # if height to width ratio is less than 1.5 skip
        if ratio < 1.2: continue 
        area = h * w
        # if width is not more than 25 pixels skip
        if width / float(w) > 25: continue #25
        # if area is less than 100 pixels skip
        #if area < 100: continue
        # draw the rectangle
        rect = cv2.rectangle(gray, (x,y), (x+w, y+h), (0,255,0),2)
        roi = thresh[y-5:y+h+5, x-5:x+w+5]
        roi = cv2.bitwise_not(roi)
        roi = cv2.medianBlur(roi, 5)
        if type(roi) is not type(None):
            text = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
        else: 
            text = "?"
        plate_num += text
    plate_num = plate_num.splitlines()
    plate_num = ''.join(map(str, plate_num))
    # use Levenshtein distance:
    dist = levenshtein_distance(character, plate_num)
    return dist, plate_num

def ocr_thresh(img, schwellenwerte, character):
    n = len(schwellenwerte)
    dist = [None] * n
    plate = [None] * n
    for i in range(n):
        dist[i], plate[i] = ocr_one(img, schwellenwerte[i], character)
        if dist[i] == 0: break 
    while None in dist: dist.remove(None)
    while None in plate: plate.remove(None)
    index_min = np.argmin(dist)
    return min(dist), plate[index_min], schwellenwerte[index_min]

def run_ocr_thresh(images, schwellenwerte, characters):
    n = len(images)
    m = len(schwellenwerte)
    dist = [0] * n
    plate = [0] * n
    wert = [0] * n
    for i in range(n):
        dist[i], plate[i], wert[i] = ocr_thresh(images[i], schwellenwerte, characters[i])
    return dist, plate, wert

dists, plates, werte = run_ocr_thresh(filenames_lpnr_gray, 
                               (70, 80, 90, 100, 110, 120, 130, 140, 150), 
                               lp_list)

