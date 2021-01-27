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


############################# UEBERLEGUNGEN ###################################
# gray_scaling auf jeden Fall drinlassen, das ist nuetzlich fuer Thresholding

# verschiedene Ansaetze beim Thresholding (aus dem OpenCV Tutorial (S.52) und 
# der zugehoerigen Dokumentation im Netz, siehe 
# https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html):
    # cv2.THRESH_BINARY (Wert fuer die Entscheidung variieren)
    # cv2.THRESH_BINARY+cv2.THRESH.OTSU (Wert ist hier 0)
    # adaptive.threshold mit mean (die blockSize und C variieren, ausserdem die 
    # Methode variieren)
    # adaptive.threshold mit gaussian (die blockSize und C variieren, ausserdem die 
    # Methode variieren)
    
# bei all diesen Ansaetzen auch einmal mit einer blur-methode (median, gauss) 
# und ohne eine durchgehen?
# image blurring (S.62ff): welchen Filter anwenden? Vor thresholding schalten?
# mehrere Filter hintereinander anwenden?

# geometrische Transformationen? viele Nummernschilder bzw. Zeichen sind schief
# im Bild. glaube Tesseract hat damit Probleme (viele Zeichen werden im 
# Tutorial unten nicht erkannt)
# also z.B. resize! verwende beide Methoden aus dem Tutorial (S.59), an einem 
# Bild habe ich keinen Unterschied feststellen koennen -> fuer mehrere Bilder
# ausprobieren und dann fuer eins entscheiden?
# Ausserdem auch affine transformation (S. 61), perspective transformation 
# (S. 61)  ausprobieren?

# morphologische Transformationen? (S.68)
# dilation wie im Tutorial unten liefert bei meinem kleinen Test bestes 
# Ergebnis, Buchstaben noch gut zu sehen, aber kleine schwarze Bereiche werden
# sehr viel kleiner.
# Trotzdem beide (erosion/dilation) ausprobieren?
# Opening auch? Closing brauchen wir wohl doch, zumindest wenn man sich das 
# OpenCV Tutorial anschaut (S.70). Opening bietet bei meinem Beispiel mit 
# rect_kern aus Tutorial unten anstelle von kernel besseres ergebnis.

# -> Reihenfolge: (gray), resize, blur (eine, mehrere, 
#    kernel-size), thresholding (binary/binary_inv, schwellenwert variieren), 
#    dilation/opening/closing!
#    warum? Tutorials machen das so, macht Sinn (wegen dimensionen der bilder)
###############################################################################
def ocr_try(img, gauss, kernel_g, med, kernel_m, both_blur, schwellenwert, 
            character):
    im = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    gray = cv2.resize(im, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
    if both_blur:
        blur_g = cv2.GaussianBlur(gray, kernel_g, 0)
        blur = cv2.medianBlur(blur_g, kernel_m)
    if gauss:
        blur = cv2.GaussianBlur(gray, kernel_g, 0)
    if med:
        blur = cv2.medianBlur(gray, kernel_m)
    ret, thresh = cv2.threshold(blur, schwellenwert, 255, cv2.THRESH_BINARY_INV)
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilation = cv2.dilate(thresh, rect_kern, iterations = 1)
    opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, rect_kern)
    #closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, rect_kern)
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
        #cv2.imshow("ROI", roi); cv2.waitKey(0)
        if type(roi) is not type(None):
            text = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
        else: 
            text = "?"
        #print(text)
        plate_num += text
    plate_num = plate_num.splitlines()
    plate_num = ''.join(map(str, plate_num))
    # use Levenshtein distance:
    dist = levenshtein_distance(character, plate_num)
    return dist

test_konturen = [None] * 437
for i in range(0, len(filenames_lpnr_gray)):
    test_konturen[i] = ocr_try(filenames_lpnr_gray[i], False, (5,5), False, 3, True, 90, lp_list[i])
    

# fuehre mehrere ocr_try fuer ein Nummernschild aus:
def run_ocr_try(image, character):
    distances = [None] * 64
    g = [(5,5), (3,3)]
    m = [3, 5]
    s = [20, 50, 90, 120, 150, 190, 220, 250]
    index = 0
    # zwei blur-Methoden, kernels und schwellenwerte durchgehen:
    for i in range(len(g)):
        for j in range(len(m)):
            for k in range(len(s)):
                distances[index] = ocr_try(img = image, gauss = False, 
                                            kernel_g = g[i], med = False, 
                                            kernel_m = m[j], both_blur = True, 
                                            schwellenwert = s[k], 
                                            character = character)
                index = index + 1
    # nur Gauss, kernels und schwellenwerte durchgehen:
    for i in range(len(g)):
        for k in range(len(s)):
            distances[index] = ocr_try(img = image, gauss = True, 
                                            kernel_g = g[i], med = False, 
                                            kernel_m = 3, both_blur = False, 
                                            schwellenwert = s[k], 
                                            character = character)
            index = index + 1
    # nur Median, kernels und schwellenwerte durchgehen:
    for j in range(len(m)):        
        for k in range(len(s)):
            distances[index] = ocr_try(img = image, gauss = False, 
                                            kernel_g = (5,5), med = True, 
                                            kernel_m = m[j], both_blur = False, 
                                            schwellenwert = s[k], 
                                            character = character)
            index = index + 1
    return distances

# mache das fuer alle nummernschilder. Ausgabe ist ein array, in dem fuer 
# jedes nummernschild eine spalte hat, in der fuer die verschiedenen preprocessing-
# methoden die distanzen ausgerechnet werden:
def grid_search(images, characters):
    n = len(images)
    res = [0] * 64
    for i in range(n):
        distances = run_ocr_try(images[i], characters[i])
        res = np.column_stack((res, distances))
    res = np.delete(res, 0, 1)
    return(res)

# mit altem tesseract:
test100 = grid_search(filenames_lpnr_gray[0:25], lp_list[0:25])
for i in range(64):
    y[i] = statistics.mean(test100[i])
    

test_all = grid_search(filenames_lpnr_gray, lp_list)

for i in range(64):
    x[i] = statistics.mean(test_all[i])
for i in range(64):
    x_med[i] = statistics.median(test_all[i])

# -> haengt hauptsaechlich vom schwellenwert ab! nutze also alle 8 fuer finale funktion?
# kernels machen keinen grossen unterschied, auch welche und ob nur zwei oder eine spielt 
# keine grosse rolle. 

# neue funktion mit beiden blur methoden und festen kernels. schwellenwert wird
# (70, 80, 90, 100, 110, 120, 130, 140, 150) variiert und bestes ergebnis wird 
# genutzt?
def ocr_one(img, schwellenwert, character):
    im = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    gray = cv2.resize(im, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
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
        #cv2.imshow("ROI", roi); cv2.waitKey(0)
        if type(roi) is not type(None):
            text = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
        else: 
            text = "?"
        #print(text)
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
    return dist, plate

def run_ocr_thresh(images, schwellenwerte, characters):
    n = len(images)
    m = len(schwellenwerte)
    res_dist = [0] * m
    res_plate = [0] * m
    for i in range(n):
        distances, plates = ocr_thresh(images[i], schwellenwerte, characters[i])
        res_dist = np.column_stack((res_dist, distances))
        res_plate = np.column_stack((res_plate, plates))
    res_dist = np.delete(res_dist, 0, 1)
    res_plate = np.delete(res_plate, 0, 1)
    return res_dist, res_plate

dists2, plates2 = run_ocr_thresh(filenames_lpnr_gray[157:162], 
                               (70, 80, 90, 100, 110, 120, 130, 140, 150), 
                               lp_list[157:162])

dist, plate = ocr_thresh("Daten_gray/eu_cars+lps/1T43213_lpnr_eu_gray.jpg", 
                         (70, 80, 90, 100, 110, 120, 130, 140, 150), "1T43213")

###############################################################################

# Levenshtein distance fuer zwei Listen (mit Levenshtein).
# Ausgabe sind drei Objekte, eine Liste mit den Distanzen, der Mittelwert und 
# der Median dieser.
def distance(list1, list2):
    distances = [None] * len(list1)
    if len(list1) != len(list2):
        print("\nerror: Listenlaengen sind unterschiedlich\n")
        return
    else:
        for i in range(0, len(list1)):
            distances[i] = levenshtein_distance(list1[i], list2[i])
    return distances, statistics.mean(distances), statistics.median(distances)


# Levenshtein mit pylev:
def distance_2(list1, list2):
    distances = [None] * len(list1)
    if len(list1) != len(list2):
        print("\nerror: Listenlaengen sind unterschiedlich\n")
        return
    else:
        for i in range(0, len(list1)):
            distances[i] = pylev.levenshtein(list1[i], list2[i])
    return distances, statistics.mean(distances), statistics.median(distances)

###############################################################################
# fuer konturen: kreiere liste, die neue namen fuer neue bilder der nummernschilder
# mit konturen enthalt:
plnr = lists.list_lpnr()
liste_lpnr_contours = lists.new_lists(plnr, "_contours")

# Tutorial von the AI Guy einfach hier reinkopiert. keine Ahnung, was die ganzen
# Funktionen so machen, aber im großen und ganzen sieht das schonmal nicht 
# schlecht aus.
# Funktion fuehrt das Nummernschildauslesen fuer unseren ganzen Datensatz aus.
def ocr_trying():
    lpnr_list = [None] * len(filenames_lpnr_gray)
    for i in range(0, len(filenames_lpnr_gray)):
        # Preprocessing:
        im = cv2.imread(filenames_lpnr_gray[i], 0) # 0 liest direkt in grayscale ein
        gray = cv2.resize(im, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
        #height, width = im.shape[:2]
        #gray1 = cv2.resize(im, (2*width, 2*height), interpolation = cv2.INTER_CUBIC)
        #cv2.imshow("gray1", gray1); cv2.waitKey(0) # kleineres ergebnis
        #cv2.imshow("gray", gray); cv2.waitKey(0)   # groesseres ergebnis
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        #cv2.imshow("gauss", blur_g); cv2.waitKey(0)
        blur = cv2.medianBlur(blur, 3)
        #cv2.imshow("median", blur_m); cv2.waitKey(0)
        # perform otsu thresh (using binary inverse since opencv contours work better with white text)
        #ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        ret2,th2 = cv2.threshold(im,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
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
        # sortiere nach den x-werten (nach x-Koordinate der linken oberen Ecke)
        # also gehe von links nach rechts ueber das Bild und sortiere die Konturen
        # nach ihren linken oberen Ecken
        sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
        
        # create copy of image
        im2 = gray.copy()
        
        #save = cv2.cvtColor(dilation, cv2.COLOR_GRAY2BGR)
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
            #cv2.imshow("ROI", roi); cv2.waitKey(0)
            if type(roi) is not type(None):
                text = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
                #rect = cv2.minAreaRect(cnt)
                #box = cv2.boxPoints(rect)
                #box = np.int0(box)
                #save = cv2.drawContours(save,[box],0,(0,150,150),2)
            else: 
                text = "?"
            #print(text)
            plate_num += text
        #print(plate_num)
        #cv2.imshow("Character's Segmented", im2); cv2.waitKey(0)
        # um Zeichenumbrueche und Leerzeichen aus den strings herauszubekommen:
        lpnr_list[i] = plate_num.splitlines()
        lpnr_list[i] = ''.join(map(str, lpnr_list[i]))
        #cv2.imwrite(liste_lpnr_contours[i], save)
    return lpnr_list


im = cv2.cvtColor(dilation, cv2.COLOR_GRAY2BGR)
for cnt in contours:
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    dilation = cv2.drawContours(dilation,[box],0,(0,150,150),2)
cv2.imwrite(liste_lpnr_contours[i], dilation)

###############################################################################

# Ausfuehren:
lpnr_liste = ocr_try()
distances, mean, median = distance(lp_list, lpnr_liste)
print(mean)
print(median)

