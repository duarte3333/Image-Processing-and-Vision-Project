import cv2
import os
import numpy as np

""" Keypoint   - Is a point of interest in the image, 
    Descriptor - A vector that describes the image patch around the keypoint
                that has 128 components, each one corresponding to the intensity 
                of a pixel in a given subregion around the keypoint """

def extract_features(video_path):
    """Extracts the features from the video and stores them in a list"""
    capture = cv2.VideoCapture(os.path.abspath(video_path))
    kp_list = []
    sift_points = [] #nome a definir no config
    sift = cv2.SIFT_create(5000) #number of sift points
    k = 0 
    while k < 4:
        success, frame = capture.read() #read the video
        if success:
            frame_points = []
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #convert image to gray
            key_points, descriptors = sift.detectAndCompute(gray,None) 
            kp_list.append(key_points)
            frame_points = ([key_points[0].pt[0],key_points[0].pt[1]]+descriptors[0].tolist())
            for i in range(1,len(key_points)):
                 temp_column = ([key_points[i].pt[0],key_points[i].pt[1]]+descriptors[i].tolist())
                 frame_points = np.column_stack((frame_points,temp_column))  
        sift_points.append(frame_points) #append everything into a list 
        k += 1
    print("(number features, nb descriptors per feature)", descriptors.shape)
    print(len(sift_points))
    return sift_points, kp_list
    
    # Framepoints:
    #  x_1  x_2 ...
    #  y_1  y_2 ...
    #  1    1
    #  .    .
    #  .    .   ...
    #  .    .
    #  128  128 ...
    
    #sift_points is a list of frame_points for each frame