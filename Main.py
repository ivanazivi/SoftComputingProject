import numpy as np
from Tkinter import *
import cv2
from PIL import Image, ImageTk

import time

from keras.models import Sequential
from keras.layers.core import Dense,Dropout, Activation, Flatten
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D, MaxPooling2D


def threshold1(image):
    value = (25, 15)
    blurred = cv2.GaussianBlur(image, value, 0)
    _, image = cv2.threshold(blurred, 0, 255,
                                    cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return image

def toYCRCB(img):
    frame1 = cv2.resize(img,(640, 480), interpolation = cv2.INTER_CUBIC)
    frame1 = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

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

def split_img(img):
    _, cr, _ = cv2.split(img)
    cr = threshold1(cr)
    return cr


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


def resize_region(region):
    return cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)


def load_video(path):
 #capture = cv2.cv.CaptureFromFile(path)
 capture = cv2.VideoCapture(path)
 pozadina = cv2.imread('Snimci/pozadina.JPG')
 frames = []
 count = int(capture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
 #count = int( cv2.cv.GetCaptureProperty(capture, cv2.cv.CV_CAP_PROP_FRAME_COUNT))

 for i in range(count):
     _, img = capture.read()
     height, width = img.shape[0:2]
     tmp = np.zeros((height,width,3), np.uint8)
     tmp = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
     tmp = subtraction(tmp, pozadina)
     tmp = split_img(tmp)
     frames.append(tmp)
 frames = np.array(frames)
 frames = frames[20:30]

 # mesavina = cv2.cv.fromarray(frames[0])
 # i = 1
 # for i in range(len(frames)):
 #     cv2.cv.AddWeighted(cv2.cv.fromarray(mesavina), 0.5, cv2.cv.fromarray(frames[i]), 0.5, 0.0, mesavina)
 # #cv2.imshow('mesavia', np.asarray(mesavina))
 # mesavina = np.asarray(mesavina)
 return frames

def find_frame_regions(frames):
    frameReg = []
    for i in range(len(frames)):
        contours, _ = cv2.findContours(frames[i], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        kontureFramova = findCountours(contours)
        kontureFramova = resize_region(kontureFramova)
        frameReg.append(kontureFramova)
    frameReg = np.array(frameReg)
    return frameReg

def matrix_to_vector(image):
    return image.flatten()

def scale_to_range(image):
    return image/255


def prepareAnn(regions):# regins niz nizova

    ready_for_ann = []
    for region in regions:
        #ready_for_ann = []
        #for r in region:
          scale = scale_to_range(region)
          ready_for_ann.append(matrix_to_vector(scale))
        #ready_for_ann2.append(ready_for_ann)
    return np.array(ready_for_ann)

def convert_output(alphabet):
    nn_outputs = []
   # nn = []
    for index in range(len(alphabet)):
    #    nn_outputs = []
     #   for i in range(len(index)):
         output = np.zeros(len(alphabet))
         output[index] = 1
         nn_outputs.append(output)
      #  nn.append(nn_outputs)
    return nn_outputs

def create_ann():
    ann = Sequential()
    ann.add(Dense(128, input_dim=1568, activation='sigmoid'))
    ann.add(Dense(70, activation='sigmoid'))

    return ann

def train_ann(ann, X_train, y_train):
    X_train = np.array(X_train) # dati ulazi
    y_train = np.array(y_train) # zeljeni izlazi za date ulaze

    # definisanje parametra algoritma za obucavanje
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)

    # obucavanje neuronske mreze
    ann.fit(X_train, y_train, nb_epoch=5000, batch_size=1, verbose = 0, shuffle=False, show_accuracy = True)

    ann.save_weights('my_model_weights.h5', overwrite=True)

    return ann

def winner(output): # output je vektor sa izlaza neuronske mreze
    return max(enumerate(output), key=lambda x: x[1])[0]

def display_result(outputs, alphabet):
    result = []
    for output in outputs:
        result.append(alphabet[winner(output)])

    counta = 0
    countb = 0
    countv = 0
    countg = 0
    countl = 0
    counto = 0
    counti = 0
    for x in result:
        if(x.startswith('a')):
            counta = counta + 1
        elif(x.startswith('b')):
            countb = countb +1
        elif(x.startswith('v')):
            countv = countv + 1
        elif(x.startswith('g')):
            countg = countg + 1
        elif(x.startswith('l')):
            countl = countl + 1
        elif(x.startswith('o')):
            counto = counto + 1
        elif(x.startswith('i')):
            counti = counti + 1

    if(counta>=6):
        vrati = "pronadjeni znak je a"
    elif(countb>=6):
        vrati = "pronadjeni znak je b"
    elif(countv>=6):
        vrati = "pronadjeni znak je v"
    elif(countg>=6):
        vrati = "pronadjeni znak je g"
    elif(countl>=6):
        vrati = "pronadjeni znak je l"
    elif(counto>=6):
        vrati = "pronadjeni znak je o"
    elif(counti>=6):
        vrati = "pronadjeni znak je i"
    else:
        vrati = "nije pronadjen nijedan znak"

    return vrati
   # return result



width, height = 800, 600
cap = cv2.VideoCapture ( 0)
cap.set ( cv2.cv.CV_CAP_PROP_FRAME_WIDTH, width)
cap.set ( cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, height)

# create the root window

root = Tk( )
root.title ( "Srpski znakovni jezik app")
root.bind ( '<Escape>', lambda e: root.quit ( ))

lmain = Label ( root)
lmain.pack ( )

#cv2.ocl.setUseOpenCL(False)
#cap = cv2.VideoCapture(0)
pozadina = readCam()
toYCRCB(pozadina)


#Treniranje NN
alphabet = ['a', 'a1', 'a2', 'a3', 'a4','a5','a6','a7', 'a8', 'a9', 'b', 'b1', 'b2', 'b3', 'b4','b5','b6','b7', 'b8', 'b9','v','v1','v2','v3','v4','v5','v6','v7','v8','v9','g','g1','g2','g3','g4','g5','g6','g7','g8','g9',
                'l','l1','l2','l3','l4','l5','l6','l7','l8','l9','o','o1','o2','o3','o4','o5','o6','o7','o8','o9','i','i1','i2','i3','i4','i5','i6','i7','i8','i9']

frameA = load_video('Snimci/ZnakA2.MP4')
frameB = load_video('Snimci/ZnakB2.MP4')
frameV = load_video('Snimci/ZnakV2.MP4')
frameG = load_video('Snimci/ZnakG.MP4')
frameI = load_video('Snimci/ZnakI2.MP4')
frameO = load_video('Snimci/ZnakO2.MP4')
frameL = load_video('Snimci/ZnakL2.MP4')
regionA = find_frame_regions(frameA)
regionB = find_frame_regions(frameB)
regionV = find_frame_regions(frameV)
regionG = find_frame_regions(frameG)
regionI = find_frame_regions(frameI)
regionO = find_frame_regions(frameO)
regionL = find_frame_regions(frameL)

regioni1 = np.concatenate((regionA, regionB))
temp = np.concatenate((regionV, regionG))
regioni2 = np.concatenate((regioni1,temp))
temp = np.concatenate((regioni2,regionL))
regioni1 = np.concatenate((temp,regionO))
temp = np.concatenate((regioni1, regionI))


input = prepareAnn(temp)
output = convert_output(alphabet)
ann = create_ann()
#ann = train_ann(ann, input, output)
ann.load_weights('my_model_weights.h5')
ann.compile(loss='categorical_crossentropy', optimizer='sgd')
result = ann.predict(np.array(input[2:4], np.float32))
print(result)
print(display_result(result, alphabet))


i=20
regionsTest = []
rez = ''
while(1):
 i = i - 1
 frameOrig = readCam()
 frame = cv2.cvtColor(frameOrig, cv2.COLOR_BGR2YCR_CB)
 fgmask = subtraction(frame, pozadina)
 cr = split_img(fgmask)

 #cv2.imshow('frameorigin',cr)

 #Pronalazenje kontura
 contours, _ = cv2.findContours(cr,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
 handContour = findCountours(contours) #originalna kontura koja nam je potrebna za prikazivanje na slici
 pom = resize_region(handContour) #resizovana kontura koja nam je potrebna za NN
 cv2.drawContours(frameOrig,[handContour],0,(0,255,0),1)
 regionsTest.append(pom)


 if(i==0):
     i = 20
     inputT = prepareAnn(regionsTest)
     result = ann.predict(np.array(inputT,np.float32))
     rez = display_result(result,alphabet)
     print(display_result(result,alphabet))
     regionsTest = []
     time.sleep(0.3)

 font = cv2.FONT_HERSHEY_SIMPLEX
 cv2.putText(frameOrig,rez,(10,560), font, 1,(255,255,255),2)
 cv2.imshow('frame1',frameOrig)


 k = cv2.waitKey(30) & 0xff
 if k == 27:
    break




cap.release()
cv2.destroyAllWindows()


