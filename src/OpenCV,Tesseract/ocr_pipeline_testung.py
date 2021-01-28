import cv2
import numpy as np
import pytesseract
from Levenshtein import distance as levenshtein_distance
import pylev
import statistics

# Setze cmd auf das Verzeichnis, in dem auch Tesseract drin ist
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Lade ausserdem settings.spydata
# Enthalten sind die Listen mit den Dateipfaden fuer die Bilder der Autos und der
# Nummernschilder je in bunt und in grau. Ausserdem ist die Liste mit den wahren
# Zeichenketten der Nummernschilder enthalten.

###############################################################################
# Fuehrt Pipeline aus. Ein Nummernschild wird ausgelesen und die Levenshtein-Distanz 
# zur wahren Zeichenkette wird berechnet.
# Input : img           = Pfad des Nummernschild-Bildes
#         schwellenwert = Wert fuer das Schwellenwertverfahren
#         character     = wahre Zeichenkette auf dem Nummernschild
# Output: dist      = Levenshtein-Distanz der erkannten Zeichenkette zur wahren
#         plate_num = erkannte Zeichenkette
def ocr_one(img, schwellenwert, character):
    # Preprocessing:
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    gray = cv2.resize(img, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
    blur_g = cv2.GaussianBlur(gray, (5,5), 0)
    blur = cv2.medianBlur(blur_g, 3)
    ret, thresh = cv2.threshold(blur, schwellenwert, 255, cv2.THRESH_BINARY_INV)
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilation = cv2.dilate(thresh, rect_kern, iterations = 1)
    opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, rect_kern)
    # Konturen:
    contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    
    plate_num = ""
    # gehe ueber Konturen und lese nur solche aus, die Zeichen sind:
    for cnt in sorted_contours:
        x,y,w,h = cv2.boundingRect(cnt)
        height, width = gray.shape
        if height / float(h) > 4: continue
        ratio = h / float(w)
        if ratio < 1.2: continue 
        if width / float(w) > 25: continue 
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
    # Levenshtein-Distanz:
    dist = levenshtein_distance(character, plate_num)
    
    return dist, plate_num


# Fuehrt Pipeline fuer mehrere Schwellenwerte aus. Dabei wird abgebrochen, sobald
# die Distanz = 0, weil dann das Nummernschild richtig ausgelesen wurde und 
# keine weiteren Schwellenwerte verwendet werden muessen.
# Input : img            = Pfad des Nummernschild-Bildes
#         schwellenwerte = Werte fuer das Schwellenwertverfahren
#         character      = wahre Zeichenkette auf dem Nummernschild
# Output: min(dist)      = minimal erreichte Distanz 
#         plate          = erkannte Zeichenkette zur minimal erreichten Distanz
#         schwellenwerte = Schwellenwert fuer die minimale Distanz
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


# Fuehre Pipeline fuer mehrere Nummernschilder und mehrere Schwellenwerte aus.
# Input : images         = Liste mit Pfaden der Nummernschild-Bilder
#         schwellenwerte = Werte fuer das Schwellenwertverfahren
#         characters     = Liste mit wahren Zeichenketten auf den Nummernschildern
# Output: dist  = minimal erreichte Distanzen fuer alle Nummernschiler (Liste)
#         plate = erkannte Zeichenketten zu den minimal erreichten Distanzen (Liste)
#         wert  = Schwellenwerte fuer die minimalen Distanzen (Liste)
def run_ocr_thresh(images, schwellenwerte, characters):
    n = len(images)
    m = len(schwellenwerte)
    dist = [0] * n
    plate = [0] * n
    wert = [0] * n
    
    for i in range(n):
        dist[i], plate[i], wert[i] = ocr_thresh(images[i], schwellenwerte, characters[i])
        
    return dist, plate, wert

###############################################################################
schwellenwerte = (70, 80, 90, 100, 110, 120, 130, 140, 150)
# Ausfuehren dauert circa 1 1/2 Stunden:
dists, plates, werte = run_ocr_thresh(filenames_lpnr_gray, 
                               (70, 80, 90, 100, 110, 120, 130, 140, 150), 
                               lp_list)

# Mittelwert und Median:
mean_dist = statistics.mean(dists)
median_dist = statistics.median(dists)

# Haeufigkeiten der Distanzen:
table_dists = [None] * (max(dists) + 1)
for i in range(max(dists) + 1):
    table_dists[i] = dists.count(i)

# Hauefigkeiten der Schwellenwerte:
table_werte = [None] * (len(schwellenwerte))
for i in range(len(schwellenwerte)):
    table_werte[i] = werte.count(schwellenwerte[i])
table_werte = np.column_stack((schwellenwerte, table_werte))
# Schwellenwert 70 wurde 145 mal gebraucht, 80 wurde 36 mal verwendet, ...

# Gespeichert ist die gesamte Ausgabe unter ocr_pipline_test.spydata