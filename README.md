# Real-time-face-recognition-using-CNN
Deep learning algorithm Convolutional neural networks with opencv has been used to design face recognition system.

Working: The real-time input image captured from camera is first fed to Viola Jones algorithm for face detection. The cropped face image is then resized to 150×150 pixels and fed to the CNN model for recognition of the class.

Dataset_gen.py : It is used to create customized dataset for the application with each class having 1000 images for training and 500 images for testing.
The Structure of the dataset :

-/dataset/

    -/train/
    
        -/1/
        
        -/2/
        
        -/3/
        
        -/4/
    -/test/
    
        -/1
                
        -/2/
        
        -/3/
        
        -/4/

Necessary packages to run:
1. Keras
2. Numpy
3. opencv-python
4. tkinter
5. time
6. os

My research paper for reference:  https://www.sciencedirect.com/science/article/pii/S1877050920311583
