import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\susan\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'

img = cv2.imread(r"C:\Users\susan\Github\license_plate_detection\src\gedrehte_schilder\FWE50_lpnr_eu_gray.jpg")



# Resize

width, height = 800, 300
imgResize = cv2.resize(img, (width, height))


# Threshold

ret, th1 = cv2.threshold(imgResize, 140, 255, cv2.THRESH_BINARY)

# Rotation

rows, columns, channels = th1.shape

R = cv2.getRotationMatrix2D((columns/2, rows/2), -15, 1)

print(R)

output = cv2.warpAffine(th1, R, (columns, rows))






# Auslesen

text = pytesseract.image_to_string(output)


#print(pytesseract.image_to_string(th1)) --- hier braucht mein Laptop mega lange
# und es gibt keine Ausgabe...woran liegt das?

houtput,woutput,_ = output.shape
boxes = pytesseract.image_to_boxes(th1)

for b in boxes.splitlines():
    #print(b)
    b = b.split(' ')
    x,y,w,h = int(b[1]),int(b[2]),int(b[3]),int(b[4])
    cv2.rectangle(output,(x,houtput-y),(w,houtput-h),(0,0,255),1)
    cv2.putText(output,b[0],(x,houtput-y+25),cv2.FONT_HERSHEY_COMPLEX,1,(50,50,255),2)

cv2.imshow("Rotiertes Bild", output)
#cv2.imshow('Schild', img)
#cv2.imshow('Vergrößert', imgResize)
cv2.imshow('Threshold', th1)


cv2.waitKey(0)
