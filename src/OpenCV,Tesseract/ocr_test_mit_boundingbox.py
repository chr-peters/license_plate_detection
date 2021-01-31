# Interessanterweise liest das Programm das Nummernschild sehr gut aus, obwohl die
# Boundingboxes noch nicht sehr gut platziert sind. Allerdings bekommt jedes Zeichen
# eine eigene Box, wobei zusätzlich noch uninteressante "Flecken" mit einbezogen werden.
# Diese werden aber anscheinend nicht zum Zeichen gezählt und es wird nur der Buchstabe/die
# Zahl ausgelesen.

# Der andere, richtige Code mit den Listen, funktioniert bei mir nicht richtig und dauert sehr lange.
# Daher habe ich hier nur eine Miniversion ohne alle Bearbeitungsschritte erstellt (lediglich
# eine Vergrößerung und das Thresholding), um die Boundingbox auszutesten.

# Erfreulich: 0 und O werden richtig erkannt!


import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\susan\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'

img = cv2.imread(r"C:\Users\susan\Spyder\AYO9034_lpnr_br_gray.jpg")



# Resize

width, height = 800, 300
imgResize = cv2.resize(img, (width, height))


# Threshold

ret, th1 = cv2.threshold(imgResize, 90, 255, cv2.THRESH_BINARY)


# Auslesen

text = pytesseract.image_to_string(th1)

#print(pytesseract.image_to_string(th1)) --- hier braucht mein Laptop mega lange
# und es gibt keine Ausgabe...woran liegt das?

hth1,wth1,_ = th1.shape
boxes = pytesseract.image_to_boxes(th1)

for b in boxes.splitlines():
    #print(b)
    b = b.split(' ')
    x,y,w,h = int(b[1]),int(b[2]),int(b[3]),int(b[4])
    cv2.rectangle(th1,(x,hth1-y),(w,hth1-h),(0,0,255),1)
    cv2.putText(th1,b[0],(x,hth1-y+25),cv2.FONT_HERSHEY_COMPLEX,1,(50,50,255),2)


cv2.imshow('Schild', img)
cv2.imshow('Vergrößert', imgResize)
cv2.imshow('Threshold', th1)


cv2.waitKey(0)

