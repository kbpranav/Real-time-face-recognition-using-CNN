# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 17:35:03 2020

@author: K B PRANAV
"""

import cv2
import matplotlib.pyplot as plt
import time
import tkinter as tk
import os
#import PIL


    
    
def init_gui():           
   window= tk.Tk()
   
   btn_arch = tk.Button(window, text="class-1", width=50, command= lambda :get_images(1))
   btn_arch.pack(anchor=tk.CENTER, expand=True)
   
   btn_arch = tk.Button(window, text="class-2", width=50, command= lambda :get_images(2))
   btn_arch.pack(anchor=tk.CENTER, expand=True)
   
   btn_arch = tk.Button(window, text="class-3", width=50, command=  lambda :get_images(3))
   btn_arch.pack(anchor=tk.CENTER, expand=True)
   
   btn_arch = tk.Button(window, text="class-4", width=50, command= lambda :get_images(4))
   btn_arch.pack(anchor=tk.CENTER, expand=True)
   
   btn_arch = tk.Button(window, text="reset", width=25, command= lambda :reset())
   btn_arch.pack(anchor=tk.CENTER, expand=True)
   
   window.mainloop()
   
def reset():
    
    for folder in ['dataset/train/1', 'dataset/train/2','dataset/train/3','dataset/train/4','dataset/test/1', 'dataset/test/2','dataset/test/3','dataset/test/4']:
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)

    print("Dataset Reset done!")
def get_images(class_num):
    
    cap1 = cv2.VideoCapture(0)
    print("Camera Starting...")
    time.sleep(1)
    #print("\nDetecting Pedestrians now...\n[INFO]Press 'q' to exit")
    count=0
    print("Capturing now.....")
    start = time.time()
    while(True):
       
        ret1,frame1 = cap1.read()
        
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        original_image = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        # Load the classifier and create a cascade object for face detection
        face_cascade = cv2.CascadeClassifier('C:\opencv\sources\samples\winrt\FaceDetection\FaceDetection\Assets\haarcascade_frontalface_alt.xml')
   
        detected_faces = face_cascade.detectMultiScale(original_image)
        #faces=[]
        
        for (column, row, width, height) in detected_faces:
            cv2.rectangle(original_image,(column, row), (column + width, row + height), (94, 102, 162), 0)
            
        if(len(detected_faces)>0):
           count+=1
        #cv2.imshow('frame',original_image)  
        crop_img = original_image[row-3:row + height+3,column-3:column+width+3]
        crop_img=cv2.resize(crop_img,(150,150))
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)           
        
        datapath="dataset/" 
        if(count<=1000):
            plt.imsave(datapath + '/train/'+str(class_num)+'/' + str(count) + '.jpg' , crop_img)
        else:
            plt.imsave(datapath + '/test/' +str(class_num)+ '/'+ str(count-1000) + '.jpg' , crop_img)
        
        
        if(count>=1500):
            break
        
    cap1.release()
    #cv2.destroyAllWindows()
    end= time.time()
    print("\n\nCapture Complete!")
    print("Time elapsed:",(end-start),"secs.")

init_gui()   
    
