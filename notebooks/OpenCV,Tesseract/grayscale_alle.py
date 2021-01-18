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

# Laenge der Listen:
len_file = len(filenames_lpnr)


# baue Listen fuer die Dateinamen fuer die grayscale-Bilder:
filenames_car_gray = [None] * len_file
for i in range(0, len_file):
    x = filenames_car[i]
    filenames_car_gray[i] = (x[0:5] + "_gray" + x[5:(len(x)-4)] + "_gray" + x[(len(x)-4):len(x)])
filenames_lpnr_gray = [None] * len_file
for i in range(0, len_file):
    x = filenames_lpnr[i]
    filenames_lpnr_gray[i] = (x[0:5] + "_gray" + x[5:(len(x)-4)] + "_gray" + x[(len(x)-4):len(x)])


# Funktion, die alle Bilder aus datenliste gray-scaled und sie in 
# datenliste_gray speichert.
# Dabei muessen Ordner Daten_gray/* schon existieren mit den vier Unterordner
# schon vorhanden sein!
def gray_scaling(datenliste, datenliste_gray):
    for i in range(0, len_file):
        img = cv2.imread(datenliste[i])
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(datenliste_gray[i], gray)


# Ausfuehren fuer Cars und Nummernschilder:
gray_scaling(filenames_lpnr, filenames_lpnr_gray)
gray_scaling(filenames_car, filenames_car_gray) 

