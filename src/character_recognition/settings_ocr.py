import glob # fuer Dateinamen aus Pfad
import xml.etree.ElementTree as ET # fuer xml-Dateien
import cv2
import numpy as np

###############################################################################
# xml-Dateien einlesen und wahre Zeichenketten der Nummernschilder herausbekommen:
    
# Funktion fuer Liste mit urspruenglichen Dateinamen der xml-Dateien der 
# Nummernschilder:
def list_xml_lpnr():
    # sammle alle Dateinamen aus den Pfaden ein:
    filenamesList_eu = glob.glob('Daten/eu_cars+lps/*_lpnr_eu.xml')
    filenamesList_br = glob.glob('Daten/br_cars+lps/*_lpnr_br.xml')
    filenamesList_ro = glob.glob('Daten/ro_cars+lps/*_lpnr_eu.xml')
    filenamesList_us = glob.glob('Daten/us_cars+lps/*_lpnr_us.xml')
    # packe alle zusammen in eine Liste:
    filenamesList = filenamesList_br + filenamesList_eu + filenamesList_ro + filenamesList_us
    return filenamesList

# Funktion, die Liste, die die Nummernschilder der Autos aus den Ordnern 
# br, eu, ro, us enthaelt (in der Reihenfolge), ausgibt:
def xml_list(xml_liste):
    lp_list = [None] * len(xml_liste) 
    for i in range(0, len(xml_liste)):
        # lese xml-Datei ein:
        tree = ET.parse(xml_liste[i])
        root = tree.getroot()
        # Finde Laenge des Nummernschildes heraus (mit NONE):
        length_lp = 0
        for object in root.iter("object"):
            length_lp = length_lp + 1
        # Schreibe die Ziffern des Nummernschildes in lp:
        lp = [None] * length_lp 
        j = 0 
        for name in root.iter("name"):
            lp[j] = name.text
            j = j + 1
        # None raus:
        while "None" in lp: lp.remove("None")
        # list zu string
        listToStr = ''.join(map(str, lp))
        # fuege die Ziffern des Nummernschildes zu lp_list hinzu:
        lp_list[i] = listToStr   
    return lp_list
 
    
# Listen mit Dateipfaden zu den Bildern der Autos und Nummernschilder erstellen:

# Funktion fuer Liste mit urspruenglichen Dateinamen der Nummernschilder:
def list_lpnr():
    # sammle alle Dateinamen aus den Pfaden ein (Nummernschilder):
    filenamesList_eu = glob.glob('Daten/eu_cars+lps/*_lpnr_eu.jpg')
    filenamesList_br = glob.glob('Daten/br_cars+lps/*_lpnr_br.jpg')
    filenamesList_ro = glob.glob('Daten/ro_cars+lps/*_lpnr_eu.jpg')
    filenamesList_us = glob.glob('Daten/us_cars+lps/*_lpnr_us.jpg')
    # packe alle zusammen in eine Liste:
    filenames_lpnr = filenamesList_br + filenamesList_eu + filenamesList_ro + filenamesList_us
    return filenames_lpnr

# Funktion fuer Liste mit urspruenglichen Dateinamen der Autos:
def list_car():
    # sammle alle Dateinamen aus den Pfaden ein (Autos):
    filenamesList_eu = glob.glob('Daten/eu_cars+lps/*_car_eu.jpg')
    filenamesList_br = glob.glob('Daten/br_cars+lps/*_car_br.jpg')
    filenamesList_ro = glob.glob('Daten/ro_cars+lps/*_car_eu.jpg')
    filenamesList_us = glob.glob('Daten/us_cars+lps/*_car_us.jpg')
    # packe alle zusammen in eine Liste:
    filenames_car = filenamesList_br + filenamesList_eu + filenamesList_ro + filenamesList_us
    return filenames_car

# Funktion fuer Listen fuer neue Dateinamen mit Zusatz neu:
# alte_liste (list) ist die zu veraendernde Liste und neu (character) ist das, 
# was in die Namen eingefuegt werden soll.
def new_lists(alte_liste, neu):
    n = len(alte_liste)
    neue_liste = [None] * n
    for i in range(0, n):
        x = alte_liste[i]
        neue_liste[i] = (x[0:5] + neu + x[5:(len(x)-4)] + neu + x[(len(x)-4):len(x)])
    return neue_liste
   
# Funktion, die alle Bilder aus datenliste gray-scaled und sie in 
# datenliste_gray speichert.
# Dabei muessen Ordner Daten_gray/* mit den vier Unterordnern schon vorhanden sein!
def gray_scaling(datenliste, datenliste_gray):
    n = len(datenliste)
    for i in range(0, n):
        img = cv2.imread(datenliste[i])
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(datenliste_gray[i], gray)

###############################################################################
# Ausfuehren fuer Autos und Nummernschilder: 
filenames_car = list_car()
filenames_lpnr = list_lpnr()

filenames_car_gray = new_lists(filenames_car, "_gray")    
filenames_lpnr_gray = new_lists(filenames_lpnr, "_gray") 
    
gray_scaling(filenames_lpnr, filenames_lpnr_gray)
gray_scaling(filenames_car, filenames_car_gray) 

# xml-Dateien:
list_xml = list_xml_lpnr()
lp_list = xml_list(list_xml)

# Gespeichert in settings.spydata