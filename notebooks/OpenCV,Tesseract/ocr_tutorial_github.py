import cv2
import numpy as np
import pytesseract
import glob # fuer Dateinamen aus Pfad


# sammle alle Dateinamen aus den Pfaden ein:
filenamesList_eu = glob.glob('Daten/eu_cars+lps/*_lpnr_eu.jpg')
filenamesList_br = glob.glob('Daten/br_cars+lps/*_lpnr_br.jpg')
filenamesList_ro = glob.glob('Daten/ro_cars+lps/*_lpnr_eu.jpg')
filenamesList_us = glob.glob('Daten/us_cars+lps/*_lpnr_us.jpg')
# packe alle zusammen in eine Liste:
filenamesList = filenamesList_br + filenamesList_eu + filenamesList_ro + filenamesList_us
# Laenge der Liste:
len_file = len(filenamesList)

# setze cmd auf das Verzeichnis, in dem auch Tesseract drin ist
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Funktion, die Tesseract ausfuehrt (nach der Funktion von dem Typen im Github):
def ocr(img):
    # grayscale region within bounding box
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # threshold the image using Otsus method to preprocess for tesseract
    #thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 11)
    # perform a median blur to smooth image slightly
    #blur = cv2.medianBlur(thresh, 3)
    # resize image to double the original size as tesseract does better with certain text size
    #blur = cv2.resize(blur, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
    # run tesseract and convert image text to string
    text = pytesseract.image_to_string(thresh, config="--psm 7", lang = "deu")
    return text

# Images einlesen und OCR ausfuehren:
lp_ocr = [None] * len_file
for i in range(0, 100):
    img = cv2.imread(filenamesList[i])
    lp_ocr[i] = ocr(img)

print(lp_ocr)




### ALT!
# Image einlesen
img = cv2.imread("Daten\JSP7678_lpnr_br.jpg")

ocr(img)
# Image verbessern
#img = cv2.resize(img, None, fx = 0.5, fy = 0.5) # resize das image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # gray-scales

adaptive_threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 11)

# lass Tesseract arbeiten, deu fuer deutsch
# die 7 steht fuer "treat the image as a single text line"
text = pytesseract.image_to_string(adaptive_threshold, config = "--psm 7", lang = "deu")
print(text)

# zeige das Image
cv2.imwrite("Output\JSP_ocr.jpg", img)
cv2.imwrite("Output\JSP_gray.jpg", gray)
cv2.imwrite("Output\JSP_thresh.jpg", thresh)
cv2.imwrite("Output\JSP_blur.jpg", blur)
