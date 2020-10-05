# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 15:17:31 2020

@author: K B PRANAV
"""

import numpy as np # linear algebra
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense,Dropout
from keras.preprocessing.image import ImageDataGenerator, img_to_array,load_img,array_to_img
from keras.callbacks import ModelCheckpoint
import time
import cv2
import tkinter as tk


class FaceRecg:
    def __init__(self):
        self.init_gui()
    
    
    def Load_model_arch(self):
        # Initialising the CNN
        self.classifier = Sequential()
        
        # Step 1 - Convolution
        self.classifier.add(Conv2D(32, (2, 2), input_shape = (150, 150, 3), activation = 'relu'))
        
        # Step 2 - Pooling
        self.classifier.add(MaxPooling2D(pool_size = (2, 2)))
        
        # Adding a second convolutional layer
        self.classifier.add(Conv2D(64, (2, 2), activation = 'relu'))
        self.classifier.add(MaxPooling2D(pool_size = (2, 2)))
        
        # Adding a third convolutional layer
        self.classifier.add(Conv2D(128, (2, 2), activation = 'relu'))
        self.classifier.add(MaxPooling2D(pool_size = (2, 2)))
        
        # Adding a fourth convolutional layer
        self.classifier.add(Conv2D(128, (2, 2), activation = 'relu'))
        self.classifier.add(MaxPooling2D(pool_size = (2, 2)))
        
                
        self.classifier.add(Flatten())
        
        self.classifier.add(Dense(units = 64, activation = 'relu'))
        self.classifier.add(Dropout(0.5))
        self.classifier.add(Dense(units = 4, activation = 'softmax'))
        
        # Compiling the CNN
        self.classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        
        print("\n[INFO]Architecture Loaded...")
    
          
    def detect_face(self):
            
        cap1 = cv2.VideoCapture(0)
        time.sleep(2)
        print("\n Starting face detection...")
        #print("\nDetecting Pedestrians now...\n[INFO]Press 'q' to exit")
        while(True):
            #name+=1
            ret1,frame1 = cap1.read()
            
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            
            original_image = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            ##cv.imshow('gIMG',grayscale_image)
        
            # Load the classifier and create a cascade object for face detection
            face_cascade = cv2.CascadeClassifier('C:\opencv\sources\samples\winrt\FaceDetection\FaceDetection\Assets\haarcascade_frontalface_alt.xml')
        
        
            detected_faces = face_cascade.detectMultiScale(original_image)
        
        
            for (column, row, width, height) in detected_faces:
                cv2.rectangle(
                    original_image,
                    (column, row),
                    (column + width, row + height),
                    (255, 0, 0),
                    2
                )
        
            print("Number of faces detected:",len(detected_faces))
            print("press 'q' to exit")
            cv2.imshow('Frame',original_image)
            key = cv2.waitKey(1)&0xFF
            if(key==ord('q')):
                break
            
        cap1.release()
        cv2.destroyAllWindows()
        
    
    def preview_ImageDataGen():
        datagen = ImageDataGenerator(
            rotation_range=40,width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
        
        img= load_img('A:\Engineering\Projects\FaceRecg\data_crop\\52.jpg')
        x=img_to_array(img)
        x=x.reshape((1,)+ x.shape)
        
        i=0
        for batch in datagen.flow(x,batch_size=1, save_to_dir='A:\Engineering\Projects\FaceRecg\preview', save_prefix='face', save_format='jpg'):
            i+=1
            if(i>10):
                break
            
    
    
    def train_model(self):
        batch_size=32
        train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2, zoom_range = 0.2,horizontal_flip = True)
                                             
        
        test_datagen = ImageDataGenerator(rescale = 1./255)
        
        training_set = train_datagen.flow_from_directory(r"A:\Engineering\Projects\FaceRecg\APP\dataset\train",target_size = (150, 150), batch_size = 32,class_mode='categorical')
       
        test_set = test_datagen.flow_from_directory(r"A:\Engineering\Projects\FaceRecg\APP\dataset\test" ,target_size = (150, 150), batch_size = 32,class_mode = 'categorical')
        
        filepath = "Trained_Weights.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        start=time.time()
        print("\n Training started.....")
        self.classifier.fit_generator(training_set,
                                 steps_per_epoch = 798//batch_size,
                                 epochs = 10,
                                 validation_data = test_set,
                                 validation_steps = 204//batch_size,
                                 callbacks = [checkpoint])
        end=time.time()
        self.classifier.save("Trained_model")
        #print(history.history.keys())
        print("\n[INFO]Trained Successfully...")
        print('Time to train:',(end-start)," seconds.")
        #return self.classifier
        
        
    def load_trained_model(self):
        #self.classifier.load_weights('best_model_Evaluation_new.hdf5')
        self.classifier.load_weights('Trained_model')
        #self.classifier.score(self.train_model().test_set)
        print("\n[INFO]Model Loaded Successfully...")
        #return self.classifier
        
    def recognize(self):
        #Initiate the camera
        cap1 = cv2.VideoCapture(0)
        print("Starting Camera...")
        face_cascade = cv2.CascadeClassifier('C:\opencv\sources\samples\winrt\FaceDetection\FaceDetection\Assets\haarcascade_frontalface_alt.xml')
        time.sleep(2)
        while(True):
            ret1,frame1 = cap1.read()
            
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            
            original_image = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            detected_faces = face_cascade.detectMultiScale(original_image)
        
            faces=[]
            for (column, row, width, height) in detected_faces:
                cv2.rectangle(original_image,(column, row),(column + width, row + height),(255, 0, 0),2)
                crop_img = original_image[row-3:row + height+3,column-3:column+width+3]
                crop_img=cv2.resize(crop_img,(150,150))
                faces.append(crop_img)
            
            for i in range(len(faces)):
                (column, row, width, height)=detected_faces[i]
                test_image = np.expand_dims(faces[i], axis = 0)
                result = self.classifier.predict(test_image)
                
                if(result[0][0]==1):
                    text="class 1"
                elif(result[0][1]==1):
                    text="class 2"
                elif(result[0][2]==1):
                    text="class 3"
                else:
                    text="class 4"
                #original_image = cv2.putText(original_image,result,(row,column))
                print("Detected class",text,result[0])
                cv2.putText(original_image,text,(column-5,row-5),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)
            cv2.imshow('Face Recognition', original_image)
            key = cv2.waitKey(1)&0xFF
            if(key==ord('q')):
                break
            
        cap1.release()
        cv2.destroyAllWindows()
                
         
    def init_gui(self):
           
            window= tk.Tk()
            
            self.btn_arch = tk.Button(window, text="Load Architecture", width=50, command=lambda:self.Load_model_arch())
            self.btn_arch.pack(anchor=tk.CENTER, expand=True)
            
            self.btn_arch = tk.Button(window, text="Check Face Detection", width=50, command=lambda:self.detect_face())
            self.btn_arch.pack(anchor=tk.CENTER, expand=True)
            
            #self.btn_arch = tk.Button(window, text="Image Data gen", width=50, command=lambda:self.preview_ImageDataGen())
            #self.btn_arch.pack(anchor=tk.CENTER, expand=True)
            
            self.btn_train = tk.Button(window ,text="Train Model", width=50, command=lambda: self.train_model())
            self.btn_train.pack(anchor=tk.CENTER, expand=True)
            
            self.btn_load = tk.Button(window,text="Load Trained Model", width=50, command=lambda: self.load_trained_model())
            self.btn_load.pack(anchor=tk.CENTER, expand=True)
            
            
            self.btn_detect = tk.Button(window,text="Recognize faces", width=50, command=lambda: self.recognize())
            self.btn_detect.pack(anchor=tk.CENTER, expand=True)
            
            window.mainloop()    
            
FaceRecg()