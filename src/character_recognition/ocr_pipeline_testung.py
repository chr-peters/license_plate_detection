import cv2
import numpy as np
import pytesseract
import pylev
import statistics

# Lade ausserdem settings.spydata
# Enthalten sind die Listen mit den Dateipfaden fuer die Bilder der Autos und der
# Nummernschilder je in bunt und in grau. Ausserdem ist die Liste mit den wahren
# Zeichenketten der Nummernschilder enthalten.

###############################################################################
# Fuehrt Pipeline aus. Ein Nummernschild wird ausgelesen und die Levenshtein-Distanz 
# zur wahren Zeichenkette wird berechnet. Mit ausgabe = True werden alle Prozesse
# im Ordner "Output/" gespeichert.
# Input : img           = Pfad des Nummernschild-Bildes
#         schwellenwert = Wert fuer das Schwellenwertverfahren
#         character     = wahre Zeichenkette auf dem Nummernschild
#         ausgabe       = true/false, sollen Bilder der Bildbearbeitung gespeichert werden?
#         name          = Name der Ausgabedatei
# Output: dist      = Levenshtein-Distanz der erkannten Zeichenkette zur wahren
#         plate_num = erkannte Zeichenkette
def ocr_one(img, schwellenwert, character, ausgabe, ausgabe_thresh, name, tes):
    # Setze cmd auf das Verzeichnis, in dem auch Tesseract drin ist
    pytesseract.pytesseract.tesseract_cmd = r"tesseract.exe"
    
    # Preprocessing:
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    gray = cv2.resize(img, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
    if ausgabe: cv2.imwrite(("Output/" + name + "_1_resize.jpg"), gray)
    blur_g = cv2.GaussianBlur(gray, (5,5), 0)
    blur = cv2.medianBlur(blur_g, 3)
    if ausgabe: cv2.imwrite(("Output/" + name + "_2_blur.jpg"), blur)
    if schwellenwert != 0:
        ret, thresh = cv2.threshold(blur, schwellenwert, 255, cv2.THRESH_BINARY_INV)
    else:
        ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    if ausgabe_thresh:  cv2.imwrite(("Output/" + name + "_3_thresh_otsu.jpg"), thresh)
    if ausgabe: cv2.imwrite(("Output/" + name + "_3_thresh_" + str(schwellenwert) + ".jpg"), thresh)
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilation = cv2.dilate(thresh, rect_kern, iterations = 1)
    if ausgabe: cv2.imwrite(("Output/" + name + "_4_dilation.jpg"), dilation)
    opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, rect_kern)
    if ausgabe: cv2.imwrite(("Output/" + name + "_5_opening.jpg"), opening)
    if tes:
        # Konturen:
        contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
        
        plate_num = ""
        i = 0
        # gehe ueber Konturen und lese nur solche aus, die Zeichen sind:
        for cnt in sorted_contours:
            x,y,w,h = cv2.boundingRect(cnt)
            height, width = gray.shape
            if height / float(h) > 4: continue
            ratio = h / float(w)
            if ratio < 1.2: continue 
            if width / float(w) > 50: continue  #25
            rect = cv2.rectangle(gray, (x,y), (x+w, y+h), (0,255,0),2)
            roi = thresh[y-5:y+h+5, x-5:x+w+5]
            roi = cv2.bitwise_not(roi)
            roi = cv2.medianBlur(roi, 5)
            if type(roi) is not type(None):
                text = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
                if ausgabe: cv2.imwrite(("Output/" + name + "_6_kontur_" + str(i) + ".jpg"), roi)
                i = i + 1
            else: continue
            plate_num += text
            
        plate_num = plate_num.splitlines()
        plate_num = ''.join(map(str, plate_num))
    else:
        plate_num = pytesseract.image_to_string(opening, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 7 --oem 3')
    # Levenshtein-Distanz:
    dist = pylev.levenshtein(character, plate_num)
    
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
def ocr_thresh(img, schwellenwerte, character, ausgabe, ausgabe_thresh, name, tes):
    if schwellenwerte != 0:
        n = len(schwellenwerte)
        dist = [None] * n
        plate = [None] * n
        for i in range(n):
            dist[i], plate[i] = ocr_one(img, schwellenwerte[i], character, ausgabe, ausgabe_thresh, name, tes)
            if dist[i] == 0: break 
        while None in dist: dist.remove(None)
        while None in plate: plate.remove(None)
        index_min = np.argmin(dist)
        return min(dist), plate[index_min], schwellenwerte[index_min]
    else:
        dist = (None)
        plate = (None)
        n = 2
        dist, plate = ocr_one(img, schwellenwerte, character, ausgabe, ausgabe_thresh, name, tes)
        return dist, plate, schwellenwerte
       
    
    


# Fuehre Pipeline fuer mehrere Nummernschilder und mehrere Schwellenwerte aus.
# Input : images         = Liste mit Pfaden der Nummernschild-Bilder
#         schwellenwerte = Werte fuer das Schwellenwertverfahren
#         characters     = Liste mit wahren Zeichenketten auf den Nummernschildern
# Output: dist  = minimal erreichte Distanzen fuer alle Nummernschiler (Liste)
#         plate = erkannte Zeichenketten zu den minimal erreichten Distanzen (Liste)
#         wert  = Schwellenwerte fuer die minimalen Distanzen (Liste)
def run_ocr_thresh(images, schwellenwerte, characters, ausgabe, ausgabe_thresh, name, tes):
    n = len(images)
    dist = [0] * n
    plate = [0] * n
    wert = [0] * n
    
    for i in range(n):
        dist[i], plate[i], wert[i] = ocr_thresh(images[i], schwellenwerte, characters[i], ausgabe, ausgabe_thresh, name, tes)
        
    return dist, plate, wert

###############################################################################
schwellenwerte = (60, 65, 70, 75, 80, 85, 90, 95, 100, 110, 120, 130, 140, 150)
# Ausfuehren dauert circa 1 1/2 Stunden:
# (vorher war es bei mir Version 4)
dists_v5, plates_v5, werte_v5 = run_ocr_thresh(filenames_lpnr_gray, 
                                               schwellenwerte, lp_list, 
                                               False, False, "bla", True)

# Mittelwert und Median:
mean_dist = statistics.mean(dists)
median_dist = statistics.median(dists)
mean_dist_v5 = statistics.mean(dists_v5)
median_dist_v5 = statistics.median(dists_v5)

# Haeufigkeiten der Distanzen:
table_dists = [None] * (max(dists) + 1)
for i in range(max(dists) + 1):
    table_dists[i] = dists.count(i)
# Das nur 38 mal das Nummernschild korrekt erkannt wurde, 70 mal nur ein Fehler
# und 86 mal zwei Fehler gemacht werden, ist relativ wenig... Ausserdem werden
# 37 mal die Schilder ueberhaupt nicht erkannt.

# Hauefigkeiten der Schwellenwerte:
table_werte = [None] * (len(schwellenwerte))
for i in range(len(schwellenwerte)):
    table_werte[i] = werte.count(schwellenwerte[i])
table_werte = np.column_stack((schwellenwerte, table_werte))
# Schwellenwert 70 wurde 145 mal gebraucht, 80 wurde 36 mal verwendet, ...
# Schwellenwerte-Vektor noch vergroessern? Mit kleinstem Schwellenwert wurde sehr 
# haeufig bestes Ergebnis erzielt. Also vielleicht noch kleinere Werte und mehr 
# Werte zwischen den 10er-Schritten hinzufuegen?
# Pruefe dazu einige spezielle Kennzeichen.

# Version 5:
# Haeufigkeiten der Distanzen:
table_dists_v5 = [None] * (max(dists_v5) + 1)
for i in range(max(dists_v5) + 1):
    table_dists_v5[i] = dists_v5.count(i)

# Hauefigkeiten der Schwellenwerte:
table_werte_v5 = [None] * (len(schwellenwerte))
for i in range(len(schwellenwerte)):
    table_werte_v5[i] = werte_v5.count(schwellenwerte[i])
table_werte_v5 = np.column_stack((schwellenwerte, table_werte_v5))

# Ueberpruefung von Otsu:
dists_o, plates_o, werte_o = run_ocr_thresh(filenames_lpnr_gray, 0, lp_list, False, False, "bla", True)

mean_dist_o = statistics.mean(dists_o)
median_dist_o = statistics.median(dists_o)
# Mittelwert von 2.99 ist ganz okay

# Ueberpruefung von Otsu mit Auslesen des gesamten Nummernschildes:
dists_og, plates_og, werte_og = run_ocr_thresh(filenames_lpnr_gray, 0, lp_list, False, False, "bla", False)
mean_dist_og = statistics.mean(dists_og)
median_dist_og = statistics.median(dists_og)
# -> funktioniert richtig schlecht!

###############################################################################
# Manuelle Ueberpruefung einzelner auffaelliger Nummernschilder:
    
# br_OKM2371: leicht schiefes Kennzeichen, zur Ueberpruefung, ob Konturen 
# erkannt werden.
print(plates[50])
print(dists[50])
# nicht wirklich schoen, schaue Konturen an:
ocr_one(filenames_lpnr[50], werte[50], lp_list[50], True, "OKM2371")
# Konturen werden nicht erkannt 
# -> Kennzeichen schlecht ausgeschnitten? Zu schraeg?

# br_PJI5921: I und 1 sind auf dem Nummernschild haargenau identisch, da kann 
# also nur ein Fehler passieren.
print(plates[93])
print(dists[93])
# die 1 hinten wird wohl gar nicht als Kontur erkannt.
ocr_one(filenames_lpnr[93], werte[93], lp_list[93], True, "PJI5921")
# sowohl das I als auch die 1 werden vom Konturen-Ueberpruefen rausgeschmissen.
# die anderen Konturen sind sehr schoen . Allerdings macht Tesseract aus 5 Konturen
# 6 Zeichen (?!)
# -> Anpassung der Bedingungen fuer Konturen? Klar, Breite der Konturen fuer
#    I und 1 ist im Vgl zur Breite des Nummernschildes viel zu klein!
# Passe haendisch die Zeile 55 an 
# if width / float(w) > 50: continue 
# und berechne erneut
ocr_one(filenames_lpnr[93], werte[93], lp_list[93], True, "PJI5921_2")
# damit werden I und 1 schonmal als Konturen erkannt und die Distanz bleibt 
# trotzdem bei 4. Behalte also die 50 erstmal bei.
# Teste OKM2371 auch nochmal mit der neuen Einstellung:
ocr_one(filenames_lpnr[50], werte[50], lp_list[50], True, "OKM2371_2")
# keine Aenderung

# eu_FWE50: leicht schiefes Kennzeichen (gleiche Problematik wie beim ersten 
# Kennzeichen?).
print(plates[120])
print(dists[120])
# nicht wirklich schoen, schaue Konturen an:
ocr_one(filenames_lpnr[120], werte[120], lp_list[120], True, "FWE50")
# auch mit 50 werden nicht mehr Konturen erkannt.
# Thresholding hat hier versagt (Wert ist bei 80). Die 0 ist nicht deutlich zu 
# sehen. Wert 75 oder 85 mal ausprobieren?
ocr_one(filenames_lpnr[120], 75, lp_list[120], True, "FWE50_75")
# schlimmer.
ocr_one(filenames_lpnr[120], 85, lp_list[120], True, "FWE50_85")
# viele Fragezeichen, mache daraus mal ein else:continue (Zeile 64) anstatt 
# else: text = "?"
ocr_one(filenames_lpnr[120], 85, lp_list[120], True, "FWE50_85")
# immer noch nicht gut, macht aus 3 Konturen 5 Zeichen...

# eu_RK878AC: car ist schief, lpnr ist gerade. Betrachte gleich mal Kennzeichen
# ausgeschnitten aus Auto-Bild.
print(plates[154])
print(dists[154])
# gerades Nummernschild liefert perfekte Uebereinstimmung.
img = cv2.imread(filenames_car[154])
# Suche in xml-Datei nach Koordinaten des Schildes und schneide aus:
img = img[167:187, 239:329]
cv2.imwrite("Output/RK878AC_plate.jpg", img)
ocr_one("Output/RK878AC_plate.jpg", werte[154], lp_list[154], True, "RK878AC")
# -> Drehung des Schildes in Betracht ziehen? Transformation von OpenCV verwenden?

# eu_WSQ3021: schwarzes Schild, weisse Schrift. 
print(plates[166])
print(dists[166])
# Distanz von 3, Q wird nicht erkannt und stattdessen ist sfl.
img = cv2.imread(filenames_car[166])
# Suche in xml-Datei nach Koordinaten des Schildes und schneide aus:
img = img[540:609, 317:620]
cv2.imwrite("Output/WSQ3021_plate.jpg", img)
ocr_one("Output/WSQ3021_plate.jpg", werte[166], lp_list[166], True, "WSQ3021")
# Distanz von 4, binary_inv ist hier ein Nachteil? Allerdings passiert wohl genau 
# dasselbe bei dem vorher verwendeten schon in Graustufen vorhandenen Bild.

# ro_B12TBI: Auto ist schief, Nummernschild ist gerade.
print(plates[168])
print(dists[168])
# gerades Nummernschild liefert perfekte Uebereinstimmung.
img = cv2.imread(filenames_car[168])
# Suche in xml-Datei nach Koordinaten des Schildes und schneide aus:
img = img[893:1139, 867:1386]
cv2.imwrite("Output/B12TBI_plate.jpg", img)
ocr_one("Output/B12TBI_plate.jpg", werte[168], lp_list[168], True, "B12TBI")
# Schwellenwert versagt. Vielleicht nen kleineren probieren?
ocr_one("Output/B12TBI_plate.jpg", 60, lp_list[168], True, "B12TBI_60")
# noch schlimmer
ocr_one("Output/B12TBI_plate.jpg", 75, lp_list[168], True, "B12TBI_75")
# besser
ocr_one("Output/B12TBI_plate.jpg", 80, lp_list[168], True, "B12TBI_80")
# wieder schlechter
# -> extrem verdrecktes Schild. 

ocr_one(filenames_lpnr[0], 0, lp_list[0], False, True, "AYO9034")
  
# Gespeichert ist die gesamte Ausgabe unter ocr_pipline_test.spydata