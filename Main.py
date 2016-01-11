import tkinter as tk
import sys
import cv2
from PIL import Image, ImageTk
import numpy as np

#promenljive
kernel = np.ones((3,3),np.uint8)

width, height = 800, 600
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


#create the root window
root = tk.Tk()
root.title("Srpski znakovni jezik app")
root.bind('<Escape>', lambda e: root.quit())

lmain = tk.Label(root)
lmain.pack()

def dilate(img):
    return cv2.dilate(img,kernel, iterations=1)

def erode(img):
    return cv2.erode(img,kernel, iterations=1)

def show_frame():
    max_area = 200
    min_area = 50
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(cv2image, (5,5),0)
    ret, show1 = cv2.threshold(blur,70,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)
    cv2.imshow('frame3', show1)

    _, contours, hierarchy = cv2.findContours(show1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        cnt=contours[i]
        area = cv2.contourArea(cnt)
        ci = 0
        if(area > max_area):
          max_area=area
          ci=i
          break

    cnt=contours[ci]

    hull = cv2.convexHull(cnt)
    drawing = np.zeros(frame.shape,np.uint8)
    cv2.drawContours(drawing,[cnt],0,(0,255,0),2)
    cv2.drawContours(drawing,[hull],0,(0,0,255),2)
    cv2.imshow('frame',drawing)

show_frame()

txt = "Izlaz iz neuronske mreze ovde"

text1 = tk.Label(root, text=txt)
text1.pack(side=tk.BOTTOM, pady=20, expand=True)

root.mainloop()