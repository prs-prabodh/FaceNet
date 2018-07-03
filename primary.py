'''
This file generates the GUI and links the facenet backend
with the database and the GUI itself
'''
import numpy as np
import cv2
import tkinter as tk
from tkinter import *
from tkinter import font
from PIL import Image, ImageTk
from facenet import *
from aligner import align

#face landmark detectors (OpenCV)
face_detector=cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
eye_detector=cv2.CascadeClassifier('cascades/haarcascade_eye.xml')

#load database from memory
database = np.load('./vitals/database.npy').item()

'''
callCounter - for drawing progress level
count - to enable/disable callCounter
name - contains identity of authenticated user
frame - contains the frame selected for authentication
'''
callCounter=0
count=False
name=''
frame=[]

#makes code a bit cleaner :D
def sq(x):
    return x*x

#makes circular frame and progress level
def Reshape(frame,x,y,r=100):
    global callCounter,count
    if (callCounter<100 and count==True):
        callCounter+=1
    for i in range(x):
        for j in range(y):
            lhs=sq(i-x/2)+sq(j-y/2)
            rhs=sq(r)
            if(lhs>rhs):
                if(lhs<rhs+1800 and (i+(callCounter*25))/285>1):
                    frame[i][j][0]=27
                    frame[i][j][1]=79
                    frame[i][j][2]=114
                    frame[i][j][3]=255
                else:
                    frame[i][j][0]=255
                    frame[i][j][1]=255
                    frame[i][j][2]=255
                    frame[i][j][3]=255
    return frame

#Set up GUI
window = tk.Tk()  #Makes main window
window.wm_title("FaceNet")
window.config(background="#FFFFFF")

#Graphics window
imageFrame = tk.Frame(window, width=400, height=900)
imageFrame.grid(row=0, column=0, padx=0, pady=2)

#Capture video frames and Reshape
lmain = tk.Label(imageFrame)
lmain.grid(row=0, column=0)
cap = cv2.VideoCapture(0)
def show_frame():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    cv2image=cv2.resize(cv2image,(400,int(480*(400/640))))
    frame=Reshape(cv2image,int(cv2image.shape[0]),int(cv2image.shape[1]))
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)


#attempts to capture a frame until face and eyes are detectable
#and selects a frame for retText() to add to the database
def addId():
    global count,frame
    count=True
    opreg.grid_forget()
    for i in range(500):
        _, frame = cap.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=face_detector.detectMultiScale(gray,1.1,5)
        eyes=[]
        if(np.any(faces)==False):
            print('No faces ',i)
            continue
        for (x,y,w,h) in faces:
            roi=gray[y:y+h,x:x+w]
            eyes=eye_detector.detectMultiScale(roi,1.3,7)
        try:
            _=eyes[1]
        except (IndexError):
            print('No eyes ',i)
            continue
        break
    ipreg.grid(row=2,column=0)
    ipregentry.focus()

#starts authentication process from GUI side
def verify():
    ipreg.grid_forget()
    global count,callCounter
    count=False
    callCounter=0
    opreg.grid(row=2,column=0,pady=(5,0))
    for i in range(500):
        _, frame = cap.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=face_detector.detectMultiScale(gray,1.1,5)
        eyes=[]
        if(np.any(faces)==False):
            print('No faces. Fail - ',i)
            continue
        for (x,y,w,h) in faces:
            roi=gray[y:y+h,x:x+w]
            eyes=eye_detector.detectMultiScale(roi,1.3,7)
        try:
            _=eyes[1]
        except (IndexError):
            print('No eyes. Fail - ',i)
            continue
        break
    name=recognizer(database,align(frame))
    opreg['text']='Identification - '+name

#adds a new identity to the database
def retText():
    global name,frame,count,callCounter,database
    name=ipregentry.get()
    print(name)
    modify_database(align(frame),name)
    database = np.load('./vitals/database.npy').item()
    ipregentry.delete(0,len(name))
    ipreg.grid_forget()
    opreg['text']='Identity Validated'
    opreg.grid(row=1,column=0,pady=(5,0))
    count=False
    callCounter=0

#clears the database. calls sync() to sync cleared database in facenet.py
def wipeOff():
    ipreg.grid_forget()
    global count,callCounter,database
    count=False
    callCounter=0
    opreg.grid(row=1,column=0,pady=(5,0))
    opreg['text']='Database deleted successfully!'
    database=dict({})
    np.save('./vitals/database.npy',database)
    database = np.load('./vitals/database.npy').item()
    sync()

#external fonts for buttons
emphasize = font.Font(family='Fira Sans', size=12, weight='bold',slant='roman')
robot = font.Font(family='Tajawal', size=10, weight='normal',slant='roman')

'''
un-comment to add user-tip

msgLabel=tk.Label(master=window,text='Center align your face and\nlook at the camera while verification',font=robot,bg='#ffffff')
msgLabel.grid(row=1,column=0,pady=(5,0))
'''

#input region(ipreg) contains label, text input area, validation button for adding identities
ipreg=tk.Frame(master=window)
ipregLabel=tk.Label(master=ipreg,text='Identity : ',bg='#ffffff',fg='#1B4F72',font=emphasize)
ipregLabel.grid(row=0,column=0)
ipregentry=tk.Entry(master=ipreg,relief='flat',width=30)
ipregentry.focus()
ipregentry.grid(row=0,column=1)
ipregbtn=tk.Button(master=ipreg,text='Validate',command=retText,relief='flat',bg='#2874A6',fg='#ffffff',activeforeground='#2874A6',activebackground='#ffffff',cursor='hand2')
ipregbtn['font']=robot
ipregbtn.grid(row=0,column=2)

#output region(opreg) contains text area to show result of various operations
opreg=tk.Label(master=window,bg='#ffffff',fg='#1B4F72',text='Identifying...',font=emphasize,wraplength=0)

#button region(btnreg) contains button to add identity, clear database and identify
btnreg=tk.Label(master=window)
btnreg.grid(row=3,column=0,pady=(5,0))

#add identity
addbutton=tk.Button(master=btnreg,text='Add Identity',command=addId,relief=FLAT,width=30,height=2,fg='#2874A6',activeforeground='#ffffff',activebackground='#2874A6',cursor='hand2')
addbutton['font']=robot
addbutton.grid(row=0,column=0,padx=50,pady=(50,10))

#clear database
rembutton=tk.Button(master=btnreg,text='Clear Database',command=wipeOff,relief=FLAT,width=30,height=2,fg='#2874A6',activeforeground='#ffffff',activebackground='#2874A6',cursor='hand2')
rembutton['font']=robot
rembutton.grid(row=1,column=0,padx=50,pady=(10,10))

#identify
authbutton=tk.Button(master=btnreg,text='Identify',command=verify,relief=FLAT,width=30,height=2,bg='#2874A6',fg='#ffffff',activeforeground='#2874A6',activebackground='#ffffff',cursor='hand2')
authbutton['font']=emphasize
authbutton.grid(row=2,column=0,padx=50,pady=(10,30))

#Tkinter loop
show_frame()  #Display 2
window.mainloop()  #Starts GUI
