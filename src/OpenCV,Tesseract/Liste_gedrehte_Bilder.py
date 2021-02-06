import glob
import xml.etree.ElementTree as ET
import cv2
import numpy as np



# Funktion


def list_lpnr_gedreht():
    filenamesList_gedreht = glob.glob('gedrehte_schilder/*_lpnr_eu_gray.jpg')
    filenames_lpnr_gedreht = filenamesList_gedreht
    return filenames_lpnr_gedreht

def list_car_gedreht():
    filenamesList_gedreht = glob.glob('gedrehte_schilder/*_car_eu_gray.jpg')
    filenames_car_gedreht = filenamesList_gedreht
    return filenames_car_gedreht



def list_xml_gedreht():
    filenamesList_gedreht_lpnr = glob.glob('gedrehte_schilder/*_lpnr_eu_gray.xml')
    filenamesList = filenamesList_gedreht_lpnr
    
    return filenamesList


def xml_list(xml_liste):
    lp_list_gedreht = [None] * len(xml_liste)
    for i in range(0, len(xml_liste)):
        tree = ET.parse(xml_liste[i])
        root = tree.getroot()
        lenght_lp = 0
        for object in root.iter("object"):
            lenght_lp = length_lp + 1
            
        lp = [None] * length_lp
        j = 0
        for name in root.iter("name"):
            lp[j] = name.text()
            j = j + 1
        while "None" in lp: lp.remove("None")
        listToStr = ''.join(map(str, lp))
        lp_list_gedreht[i] = listToStr
    return lp_list_gedreht
