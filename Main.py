import numpy as np
import cv2
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD



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

def resize_frame(frame):
    return cv2.resize(frame, (128,128), interpolation=cv2.INTER_NEAREST)

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
 frames = frames[10:30]
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


def prepareAnn(regions):
    ready_for_ann = []
    for region in regions:
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))
    return np.array(ready_for_ann)

def convert_output(alphabet):
    nn_outputs = []
    for index in range(len(alphabet)):
        output = np.zeros(len(alphabet))
        output[index] = 1
        nn_outputs.append(output)
    return nn_outputs

def create_ann():
    ann = Sequential()
    ann.add(Dense(128, input_dim=1568, activation='sigmoid'))
    ann.add(Dense(20, activation='sigmoid'))
    return ann

def train_ann(ann, X_train, y_train):
    X_train = np.array(X_train) # dati ulazi
    y_train = np.array(y_train) # zeljeni izlazi za date ulaze

    # definisanje parametra algoritma za obucavanje
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)

    # obucavanje neuronske mreze
    ann.fit(X_train, y_train, nb_epoch=2000, batch_size=1, verbose = 0, shuffle=False, show_accuracy = False)

    return ann

def winner(output): # output je vektor sa izlaza neuronske mreze
    return max(enumerate(output), key=lambda x: x[1])[0]

def display_result(outputs, alphabet):
    result = []
    for output in outputs:
        result.append(alphabet[winner(output)])
    return result

#cv2.ocl.setUseOpenCL(False)
cap = cv2.VideoCapture(0)
pozadina = readCam()
toYCRCB(pozadina)

alphabet = ['a', 'a1', 'a2', 'a3', 'a4','a5','a6','a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15', 'a16', 'a17', 'a18', 'a19']
frameA = load_video('Snimci/ZnakA.MP4')
regionA = find_frame_regions(frameA)

input = prepareAnn(regionA)
output = convert_output(alphabet)
ann = create_ann()
ann = train_ann(ann, input, output)

result = ann.predict(np.array(input[2:4], np.float32))
print(result)
print(display_result(result, alphabet))


i=20
regionsTest = []
while(1):
 i = i - 1
 frameOrig = readCam()
 frame = cv2.cvtColor(frameOrig, cv2.COLOR_BGR2YCR_CB)
 fgmask = subtraction(frame, pozadina)
 cr = split_img(fgmask)

 cv2.imshow('frameorigin',cr)

 #Pronalazenje kontura
 contours, _ = cv2.findContours(cr,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
 handContour = findCountours(contours) #originalna kontura koja nam je potrebna za prikazivanje na slici
 pom = resize_region(handContour) #resizovana kontura koja nam je potrebna za NN
 cv2.drawContours(frameOrig,[handContour],0,(0,255,0),1)
 regionsTest.append(pom)

 if(i==0):
     i = 20
     inputT = prepareAnn(regionsTest)
     result = ann.predict(np.array(inputT))
     print(display_result(result,alphabet))
     regionsTest = []

 cv2.imshow('frame1',frameOrig)

 k = cv2.waitKey(30) & 0xff
 if k == 27:
    break

cap.release()
cv2.destroyAllWindows()