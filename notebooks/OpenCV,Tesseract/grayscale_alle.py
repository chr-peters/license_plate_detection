import glob # fuer Dateinamen aus Pfad
import cv2
import numpy as np


# sammle alle Dateinamen aus den Pfaden ein (Nummernschilder):
filenamesList_eu = glob.glob('Daten/eu_cars+lps/*_lpnr_eu.jpg')
filenamesList_br = glob.glob('Daten/br_cars+lps/*_lpnr_br.jpg')
filenamesList_ro = glob.glob('Daten/ro_cars+lps/*_lpnr_eu.jpg')
filenamesList_us = glob.glob('Daten/us_cars+lps/*_lpnr_us.jpg')
# packe alle zusammen in eine Liste:
filenames_lpnr = filenamesList_br + filenamesList_eu + filenamesList_ro + filenamesList_us

# sammle alle Dateinamen aus den Pfaden ein (Autos):
filenamesList_eu = glob.glob('Daten/eu_cars+lps/*_car_eu.jpg')
filenamesList_br = glob.glob('Daten/br_cars+lps/*_car_br.jpg')
filenamesList_ro = glob.glob('Daten/ro_cars+lps/*_car_eu.jpg')
filenamesList_us = glob.glob('Daten/us_cars+lps/*_car_us.jpg')
# packe alle zusammen in eine Liste:
filenames_car = filenamesList_br + filenamesList_eu + filenamesList_ro + filenamesList_us


# Funktion fuer Listen fuer die Dateinamen fuer die grayscale-Bilder:
# alte_liste (list) ist die zu veraendernde Liste und neu (character) ist das, 
# was in die Namen eingefuegt werden soll.
def new_lists(alte_liste, neu):
    n = len(alte_liste)
    neue_liste = [None] * n
    for i in range(0, n):
        x = alte_liste[i]
        neue_liste[i] = (x[0:5] + neu + x[5:(len(x)-4)] + neu + x[(len(x)-4):len(x)])
    return neue_liste
   
# Ausfuehren fuer Autos und Nummernschilder: 
filenames_car_gray = new_lists(filenames_car, "_gray")    
filenames_lpnr_gray = new_lists(filenames_lpnr, "_gray") 
    

# Funktion, die alle Bilder aus datenliste gray-scaled und sie in 
# datenliste_gray speichert.
# Dabei muessen Ordner Daten_gray/* mit den vier Unterordnern schon vorhanden sein!
def gray_scaling(datenliste, datenliste_gray):
    n = len(datenliste)
    for i in range(0, n):
        img = cv2.imread(datenliste[i])
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(datenliste_gray[i], gray)


# Ausfuehren fuer Autos und Nummernschilder:
gray_scaling(filenames_lpnr, filenames_lpnr_gray)
gray_scaling(filenames_car, filenames_car_gray) 
