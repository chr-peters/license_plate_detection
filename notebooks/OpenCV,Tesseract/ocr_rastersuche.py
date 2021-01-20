import cv2
import numpy as np
import pytesseract
from Levenshtein import distance as levenshtein_distance

# setze cmd auf das Verzeichnis, in dem auch Tesseract drin ist
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# lade ausserdem Dateinamen_Listen.spydata und xml_nummernschilder_einlesen.spydata
# per drag&drop in den Variable explorer!


# Tutorial von the AI Guy einfach hier reinkopiert. keine Ahnung, was die ganzen
# Funktionen so machen, aber im großen und ganzen sieht das schonmal nicht 
# schlecht aus.
# Funktion fuehrt das Nummernschildauslesen fuer unseren ganzen Datensatz aus.
def ocr_try():
    lpnr_list = [None] * len(filenames_lpnr_gray)
    for i in range(0, len(filenames_lpnr_gray)):
        # Preprocessing:
        gray = cv2.imread(filenames_lpnr_gray[i], 0)
        gray = cv2.imread("Daten_gray/br_cars+lps\ODJ1599_lpnr_br_gray.jpg", 0)
        gray = cv2.resize( gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
        #cv2.imshow("gray", gray); cv2.waitKey(0)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        #cv2.imshow("gray", blur); cv2.waitKey(0)
        gray = cv2.medianBlur(gray, 3)
        #cv2.imshow("gray", gray); cv2.waitKey(0)
        # perform otsu thresh (using binary inverse since opencv contours work better with white text)
        ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        #cv2.imshow("Otsu", thresh); cv2.waitKey(0)
        rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        # apply dilation 
        dilation = cv2.dilate(thresh, rect_kern, iterations = 1)
        #cv2.imshow("dilation", dilation); cv2.waitKey(0)
        # find contours
        try:
            contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        except:
            ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
        
        # create copy of image
        im2 = gray.copy()
        
        plate_num = ""
        # loop through contours and find letters in license plate
        for cnt in sorted_contours:
            x,y,w,h = cv2.boundingRect(cnt)
            height, width = im2.shape
            # if height of box is not a quarter of total height then skip
            if height / float(h) > 6: continue
            ratio = h / float(w)
            # if height to width ratio is less than 1.5 skip
            if ratio < 1.5: continue
            area = h * w
            # if width is not more than 25 pixels skip
            if width / float(w) > 15: continue
            # if area is less than 100 pixels skip
            if area < 100: continue
            # draw the rectangle
            rect = cv2.rectangle(im2, (x,y), (x+w, y+h), (0,255,0),2)
            roi = thresh[y-5:y+h+5, x-5:x+w+5]
            roi = cv2.bitwise_not(roi)
            roi = cv2.medianBlur(roi, 5)
            cv2.imshow("ROI", roi); cv2.waitKey(0)
            text = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
            #print(text)
            plate_num += text
        #print(plate_num)
        #cv2.imshow("Character's Segmented", im2); cv2.waitKey(0)
        # um Zeichenumbrueche und Leerzeichen aus den strings herauszubekommen:
        lpnr_list[i] = plate_num.splitlines()
        lpnr_list[i] = ''.join(map(str, lpnr_list[i]))
    return lpnr_list


# Levenshtein distance fuer zwei Listen:
def distance(list1, list2):
    distances = [None] * len(list1)
    if len(list1) != len(list2):
        print("\nerror: Listenlaengen sind unterschiedlich\n")
        return
    else:
        for i in range(0, len(list1)):
            distances[i] = levenshtein_distance(list1[i], list2[i])
    return distances


# Ausfuehren:
lpnr_liste = ocr_try() # noch error,
# Vermutung: das objekt roi (wird in zeile 59 erzeugt) ist ein nonetype object
#            und das kann tesseract natuerlich nicht auslesen. 
# Warum ist es ein nonetype object? die ganzen schritte der bildbearbeitung 
# bringen uns nicht zum ziel, die einzelnen buchstaben in konturen auszugeben, 
# also sind gar keine konturen zu finden. Bspw bei Bild ODJ1599_lpnr_br_gray.jpg
# passiert genau das.
distance(lp_list[0:40], lpnr_liste)






