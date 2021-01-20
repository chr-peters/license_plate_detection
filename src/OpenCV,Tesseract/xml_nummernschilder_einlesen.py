import xml.etree.ElementTree as ET # fuer xml-Dateien
import glob # fuer Dateinamen aus Pfad


# sammle alle Dateinamen aus den Pfaden ein:
filenamesList_eu = glob.glob('Daten/eu_cars+lps/*_lpnr_eu.xml')
filenamesList_br = glob.glob('Daten/br_cars+lps/*_lpnr_br.xml')
filenamesList_ro = glob.glob('Daten/ro_cars+lps/*_lpnr_eu.xml')
filenamesList_us = glob.glob('Daten/us_cars+lps/*_lpnr_us.xml')
# packe alle zusammen in eine Liste:
filenamesList = filenamesList_br + filenamesList_eu + filenamesList_ro + filenamesList_us
# Laenge der Liste:
len_file = len(filenamesList)


# baue Liste, die die Nummernschilder der Autos aus den Ordners br, eu, ro, us 
# enthaelt (in der Reihenfolge):
lp_list = [None] * len_file 
for i in range(0, len_file):
    # lese xml-Datei ein:
    tree = ET.parse(filenamesList[i])
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
 



# Hier nochmal iwann die Boxen der Ziffern herausfinden: 
for bndbox in root.iter("bndbox"):
    print(bndbox.attrib)
