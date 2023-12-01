import cv2
import os
import numpy as np

""" Keypoint   - Is a point of interest in the image, 
    Descriptor - A vector that describes the image patch around the keypoint
                that has 128 components, each one corresponding to the intensity 
                of a pixel in a given subregion around the keypoint """
def count_frames(video_path):
    """Displays the video and counts the number of frames"""
    capture = cv2.VideoCapture(os.path.abspath(video_path))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frames of the video: ", total_frames)
    
def extract_features(video_path):
    """Extracts the features from the video and stores them in a list"""
    capture = cv2.VideoCapture(os.path.abspath(video_path))
    kp_list = []
    sift_points = [] #nome a definir no config
    sift = cv2.SIFT_create(5000) #number of sift points
    img1, img2 = None, None
    k = 0
    count_frames(video_path)
    while k <= 1900:
        capture.set(cv2.CAP_PROP_POS_FRAMES, k)
        success, frame = capture.read() #read the video
        if success:
            if (k == 0):
                img1 = frame
            if (k == 1900):
                img2 = frame
            frame_points = []
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #convert image to gray
            key_points, descriptors = sift.detectAndCompute(gray,None) 
            kp_list.append(key_points)
            frame_points = ([key_points[0].pt[0],key_points[0].pt[1]]+descriptors[0].tolist())
            for i in range(1,len(key_points)):
                 temp_column = ([key_points[i].pt[0],key_points[i].pt[1]]+descriptors[i].tolist())
                 frame_points = np.column_stack((frame_points,temp_column))  
        sift_points.append(frame_points) #append everything into a list 
        k += 100
    print("(Nº features, Nº descriptors per feature): ", descriptors.shape)
    print("Nº of frames extracted: ", len(sift_points))
    return sift_points, kp_list, img1, img2
    
    # Framepoints:
    #  x_1  x_2 ...
    #  y_1  y_2 ...
    #  1    1
    #  .    .
    #  .    .   ...
    #  .    .
    #  128  128 ...
    
    #sift_points is a list of frame_points for each frame