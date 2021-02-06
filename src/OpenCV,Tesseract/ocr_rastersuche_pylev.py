import cv2
import numpy as np
#np.load(r"C:\Users\susan\Github\license_plate_detection\src\OpenCV,Tesseract", allow_pickle = True)
import pytesseract
#from Levenshtein import distance as levenshtein_distance
import pylev
import statistics

# setze cmd auf das Verzeichnis, in dem auch Tesseract drin ist
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\susan\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# Lade ausserdem settings.spydata
# Enthalten sind die Listen mit den Dateipfaden fuer die Bilder der Autos und der
# Nummernschilder je in bunt und in grau. Ausserdem ist die Liste mit den wahren
# Zeichenketten der Nummernschilder enthalten.

###############################################################################
# Tutorial vom AI Guy:
# Funktion fuehrt das Nummernschildauslesen fuer gesamten Datensatz aus.
def ocr_trying():
    lpnr_list = [None] * len(filenames_lpnr_gray)
    for i in range(0, len(filenames_lpnr_gray)):
        # Preprocessing:
        im = cv2.imread(filenames_lpnr_gray[i], 0) # 0 liest direkt in grayscale ein
        gray = cv2.resize(im, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
        #cv2.imshow("gray1", gray1); cv2.waitKey(0) # kleineres ergebnis
        #cv2.imshow("gray", gray); cv2.waitKey(0)   # groesseres ergebnis
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        #cv2.imshow("gauss", blur_g); cv2.waitKey(0)
        blur = cv2.medianBlur(blur, 3)
        #cv2.imshow("median", blur_m); cv2.waitKey(0)
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
            cv2.imshow("ROI", roi); cv2.waitKey(0)
            if type(roi) is not type(None):
                text = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
            else: 
                text = "?"
            plate_num += text
        # um Zeichenumbrueche und Leerzeichen aus den strings herauszubekommen:
        lpnr_list[i] = plate_num.splitlines()
        lpnr_list[i] = ''.join(map(str, lpnr_list[i]))
    return lpnr_list

###############################################################################
# Daran angelehnt fuehre Rastersuche durch:

# Funktion fuehrt Pipeline fuer ein Nummernschild durch, dabei koennen Schritte
# und Parameter variiert werden.
# Input: img           = Dateipfad zum Bild 
#        gauss         = true/false, soll nur Gaussfilter verwendet werden?
#        kernel_g      = Kernel des Gaussfilters (x,y)
#        med           = true/false, soll nur Medianfilter verwendet werden?
#        kernel_m      = Kernel des Medianfilter 
#        both_blur     = true/false, sollen Gauss- und Medianfilter hintereinander
#                        angewendet werden?
#        schwellenwert = Wert fuer das Schwellenwertverfahren
#        character     = wahre Zeichenkette des Nummernschildes
# Output: dist = Levenshtein-Distanz des ausgelesenen Nummernschilds zur wahren
#                Zeichenfolge
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
    
    #rotation 
    rows, columns, channels = opening.shape
    R = cv2.getRotationMatrix2D((columns/2, rows/2), -15, 1)

    print(R)

    output = cv2.warpAffine(opening, R, (columns, rows))

    contours, hierarchy = cv2.findContours(output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    plate_num = ""
    
    for cnt in sorted_contours:
        x,y,w,h = cv2.boundingRect(cnt)
        height, width = gray.shape
        if height / float(h) > 4: continue #6
        ratio = h / float(w)
        if ratio < 1.2: continue 
        if width / float(w) > 25: continue #25
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
    dist = pylev.levenshtein(character, plate_num)
    return dist


# Funktion fuehrt 64 verschiedene Vorverarbeitungsprozesse fuer ein 
# Nummernschild aus.
# Input: image     = Dateipfad des Bildes
#        character = wahre Zeichenkette
# Output: distances = Liste mit 64 Distanzen fuer die verschiedenen Prozesse
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


# Funktion fuehrt die Funktion run_ocr_try fuer mehrere Nummernschilder aus.
# Input: images     = Liste mit Dateipfaden der Bilder
#        characters = Liste mit wahren Zeichenketten
# Output: res = Matrix (64x437), in den Spalten sind die Nummernschilder und in
#               den Zeilen die verschiedenen Prozesse
def grid_search(images, characters):
    n = len(images)
    res = [0] * 64
    for i in range(n):
        distances = run_ocr_try(images[i], characters[i])
        res = np.column_stack((res, distances))
    res = np.delete(res, 0, 1)
    return res


# Gehe fuer alle Nummernschilder des Datensatzes die 64 verschiedenen Prozesse
# durch (dauert lange! 4 1/2 Stunden circa).
#test_all = grid_search(filenames_lpnr_gray, lp_list)

# Berechne Mittelwert und Median der Prozesse
x = [None] * 64
x_med = [None] * 64
for i in range(64):
    x[i] = statistics.mean(test_all[i])
for i in range(64):
    x_med[i] = statistics.median(test_all[i])

# Ergebnis:
# Mittelwert und Median sind von allen Prozessen nicht gut. Meist erreichen wir 
# eine Distanz von 5-6, was nicht wirklich gut ist.
# Betrachtet man test_all, so erkennt man in den Spalten eine gewisse Struktur.
# Es laesst sich erkennen, dass mit bestimmten Schwellenwerten die Nummernschilder
# besser ausgelesen werden als mit anderen, unabhaengig von den Filtermethoden 
# und den Kernels. 
# Daraus schliessen wir, dass die Fiktermethoden und die Kernels fuer das
# Auslesen der Nummernschilder vernachlaessigt werden koennen und wir uns auf 
# die Schwellenwerte konzentrieren. Dazu nehme feste Filtermethoden und varriere
# nur noch die Schwellenwerte.

###############################################################################
# Nutze also beide Filtermethoden mit festen Kernels und variiere den 
# Schwellenwert:

# Fuehrt Pipeline aus. Ein Nummernschild wird ausgelesen und die Levenshtein-Distanz 
# zur wahren Zeichenkette wird berechnet.
# Input : img           = Pfad des Nummernschild-Bildes
#         schwellenwert = Wert fuer das Schwellenwertverfahren
#         character     = wahre Zeichenkette auf dem Nummernschild
# Output: dist      = Levenshtein-Distanz der erkannten Zeichenkette zur wahren
#         plate_num = erkannte Zeichenkette
def ocr_one(img, schwellenwert, character):
    im = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    gray = cv2.resize(im, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
    blur_g = cv2.GaussianBlur(gray, (5,5), 0)
    blur = cv2.medianBlur(blur_g, 3)
    ret, thresh = cv2.threshold(blur, schwellenwert, 255, cv2.THRESH_BINARY_INV)
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilation = cv2.dilate(thresh, rect_kern, iterations = 1)
    opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, rect_kern)

    contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    plate_num = ""
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
    dist = pylev.levenshtein(character, plate_num)
    
    return dist, plate_num

# Fuehrt Pipeline fuer mehrere Schwellenwerte aus. Dabei wird abgebrochen, sobald
# die Distanz = 0, weil dann das Nummernschild richtig ausgelesen wurde und 
# keine weiteren Schwellenwerte verwendet werden muessen.
# Input : img            = Pfad des Nummernschild-Bildes
#         schwellenwerte = Werte fuer das Schwellenwertverfahren
#         character      = wahre Zeichenkette auf dem Nummernschild
# Output: dist  = Liste der Distanzen fuer jeden Schwellenwert
#         plate = Liste der erkannten Zeichenketten
def ocr_thresh(img, schwellenwerte, character):
    n = len(schwellenwerte)
    dist = [None] * n
    plate = [None] * n
    for i in range(n):
        dist[i], plate[i] = ocr_one(img, schwellenwerte[i], character)
        if dist[i] == 0: break 
    return dist, plate

# Fuehre Pipeline fuer mehrere Nummernschilder und mehrere Schwellenwerte aus.
# Input : images         = Liste mit Pfaden der Nummernschild-Bilder
#         schwellenwerte = Werte fuer das Schwellenwertverfahren
#         characters     = Liste mit wahren Zeichenketten auf den Nummernschildern
# Output: res_dist  = Distanzen fuer alle Nummernschiler (Listen)
#         res_plate = erkannte Zeichenketten (Listen)
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


# Fuehre das mal fuer ein Beispielnummernschild aus:
dist, plate = ocr_thresh("Daten_gray/eu_cars+lps/1T43213_lpnr_eu_gray.jpg", 
                         (70, 80, 90, 100, 110, 120, 130, 140, 150), "1T43213")

# Zu sehen ist, dass die Funktion nach dem fuenften Durchlauf abgebrochen hat,
# weil das Nummernschild schon richtig ausgelesen wurde. Fuer die anderen 
# Durchlaeufe sind die Distanzen und die Zeichen auch ausgegeben.

# Fuehre Pipeline fuer zwei Nummernschilder aus (Berechnungen fuer mehr dauern 
# relativ lange): 
dists2, plates2 = run_ocr_thresh(filenames_lpnr_gray[157:159], 
                               (70, 80, 90, 100, 110, 120, 130, 140, 150), 
                               lp_list[157:159])
# Fuer das erste Nummernschild konnte nie eine exakte Uebereinstimmung erzeugt
# werden, allerdings wird zweimal nur ein Fehler gemacht. Fuer das zweite Schild
# wird schon nach beim zweiten Durchlauf das Nummernschild richtig ausgelesen,
# da wird also gestoppt.
# Beim ersten Schild kann man erkennen, dass der Schwellenwert sehr wichtig ist,
# da die Differenz je nach Schwellenwert 1 bis 7 betraegt.

# Gespeichert in ocr_rastersuche.spydata
