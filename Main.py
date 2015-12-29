
from tkinter import *
from ttk import Frame, Button, Style
import sys
import cv2
# create the root window
#root = Tk()


# modify the window
#root.title("Srpski znakovni jezik")
#root.geometry("1200x600")
#self.style = Style()
#self.style.theme_use("default")
#txt = "Izlaz iz neuronske mreze ovde"
#text1 = Label(root, text=txt)

#frame = Frame(self, relief=RAISED,  borderwidth=2)
#frame.pack(expand=True)

#text1.pack(side=BOTTOM, pady=20, expand=True)

# Start the window's event-loop
#root.mainloop()


#set video source to the default webcam
cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()