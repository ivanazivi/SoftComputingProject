import numpy as np
import cv2

def threshold0(image):
    value = (3, 3)
    blurred = cv2.GaussianBlur(image, value, 0)
    _, image = cv2.threshold(blurred, 0, 255,
                                    cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return image

def threshold1(image):
    value = (25, 15)
    blurred = cv2.GaussianBlur(image, value, 0)
    _, image = cv2.threshold(blurred, 0, 255,
                                    cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return image

def threshold2(image):
    value = (25, 25)
    blurred = cv2.GaussianBlur(image, value, 0)
    _, image = cv2.threshold(blurred, 0, 255,
                                    cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return image

def subtraction(img, img1):
    image = np.absolute(img1 - img) #pozadina - frame
    image = image >= 0
    image = image.astype('uint8')
    image = image * img #sa frame se mnozilo

    return image
def readCam():
    _, frame = cap.read()
    frame = cv2.flip(frame,1)

    return frame

def toYCRCB(img):
    frame1 = cv2.resize(img,(640, 480), interpolation = cv2.INTER_CUBIC)
    frame1 = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

def findCountours(contours):
 ci = 0
 max_area = 5000
 for i in range(len(contours)):
    cnt=contours[i]
    area = cv2.contourArea(cnt)

    if(area > max_area):
      max_area=area
      ci=i
      break



 cnt=contours[ci]
 #drawing = np.zeros(frameOrig.shape,np.uint8)
 realHandLen = cv2.arcLength(cnt, True)
 handContour = cv2.approxPolyDP(cnt, 0.001 * realHandLen, True)

 return handContour

cv2.ocl.setUseOpenCL(False)
cap = cv2.VideoCapture(0)
pozadina = readCam()
toYCRCB(pozadina)

while(1):
 frameOrig = readCam()
 frame = cv2.cvtColor(frameOrig, cv2.COLOR_BGR2YCrCb)

 fgmask = subtraction(frame, pozadina)

 _, cr, _ = cv2.split(fgmask)

 cr = threshold1(cr)
 cv2.imshow('frameorigin',cr)

 #Pronalazenje kontura
 _, contours, hierarchy = cv2.findContours(cr,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
 handContour = findCountours(contours)
 cv2.drawContours(frameOrig,[handContour],0,(0,255,0),1)

 cv2.imshow('frame1',frameOrig)

 k = cv2.waitKey(30) & 0xff
 if k == 27:
    break

cap.release()
cv2.destroyAllWindows()