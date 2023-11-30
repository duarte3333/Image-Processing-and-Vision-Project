from numpy.linalg import eig
import numpy as np
import cv2
import os
import sys
from src.extract_features import *
from src.matching_features import *
from src.homography import *
from src.ransac import *
from src.parsing import *

# OVERVIEW:
#   Feature detection: opencv
#   Matching : sklearn , numpy
#   Create Homography: numpy
#   RANSAC: numpy
    
def display_video(video_path):
    #Display the video
    capture = cv2.VideoCapture(os.path.abspath(video_path))
    while(capture.isOpened()):
        ret, frame = capture.read()
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  ##press q if you want the video to stop 
            break
    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    if (check_syntax()):
        sys.exit(1)
    config_data = parse_configuration_file(sys.argv[1])
    video_path = config_data[0].split(' ')[1].strip()
    display_video(video_path)
    sift_points, kp_list = extract_features(video_path)
    match  = matching_features(sift_points)
    src, dst, eigenValues  =  create_homography(match, kp_list)
    H, inliers = RANSAC(src, dst, 500, 0.8, eigenValues)
    print('condition:',np.linalg.cond(H),'inliers: ', inliers)
    
# Condition -> A*A^(-1) - High condition means small changes in the input can result in large changes in the output