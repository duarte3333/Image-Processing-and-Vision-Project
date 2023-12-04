from numpy.linalg import eig
import numpy as np
import cv2
import os
import sys
import pickle
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
    """ Display the video """
    capture = cv2.VideoCapture(os.path.abspath(video_path))
    if not capture.isOpened():
        print("Error: Unable to open video file.")
        return
    while True:
        ret, frame = capture.read()
        if not ret:
            print("Error: Failed to read frame.")
            break
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()

def homography_two_frames(img1, img2, sift_points, kp_list, option):
    """ Compute the homography for two frames """
    #match = matching_features(sift_points, img1, img2, cv2.SIFT_create(5000))
    match = matching_features_matrix(sift_points)
    if (option == 1):
        src, dst, H  = create_homography_openCV(match, kp_list, 0, 19)
    if (option == 2):
        src, dst, H  = create_homography(match, kp_list, 0, 19)
        H, inliers = RANSAC(src, dst, 50, 0.8)
    test_homography(img1, img2, H)
    
def create_all_homographies(match, kp_list):
    """ Compute the homography for all frames """
    i = 0
    size = len(match)
    matrix_H = [[None for _ in range(size)] for _ in range(size)] 
    while (match[i]):
        j = 0
        while (match[j]):
            print(match[i][j])
            matrix_H[i][j] = create_homography(match[i][j], kp_list, i, j)
            j += 1
        i += 1
    return matrix_H
    
def parse_points(config_data):
    """ Parse the points from the configuration file """
    line_map = config_data[1].split('   ')
    print(line_map)
    line_frame = config_data[2].split('   ')
    i = 2
    match_img = []
    match_map = []
    size = len(line_map)
    while (size > i):
        #if (i+1 < size):
        match_img.append((line_map[i].strip(), line_map[i+1].strip()))
        match_map.append((line_map[i].strip(), line_map[i+1].strip()))
        i+=2
    print("image matches: ", match_img)
    print("map matches: ", match_map)
    # return match_img, match_map
    # match_img_map = []
    # for i in range(1, len(config_data)):
    #     match_img_map.append(config_data[i].split(' ')[1].strip())
    # return match_img_map

if __name__ == "__main__":

    if (check_syntax()):
        sys.exit(1)
    config_data = parse_configuration_file(sys.argv[1])
    match_img_map = parse_points(config_data)
    video_path = config_data[0].split(' ')[1].strip()
    #display_video(video_path)
    
    sift_points, kp_list, img1, img2 = extract_features(video_path)
    
    homography_two_frames(img1, img2, sift_points, kp_list, 1) #option 1 - with openCV; option 2 - with numpy
    
"""    match2 = matching_features_SCIKITLEARN(sift_points)
    print(match2)
    
    matrix_H = create_all_homographies(match2, kp_list)
    output_file_path = 'path/file_for_transforms.ext'
    with open(output_file_path, 'wb') as file:
        pickle.dump(matrix_H, file) """
    
# Condition -> A*A^(-1) - High condition means small changes in the input can result in large changes in the output


#WHAT IS MATCH?
#match = {[[D_matchFrame1WithFrame2_1, .. DmatchFrame1WithFrame2_N], [D_matchFrame1WithFrameN, .. DmatchFrame1WithFrameN]],
#                                                   .....
#        [[D_matchFrameNWithFrame1_1, .. DmatchFrameNWithFrame1_N], [D_matchFrameNWithFrameN_1, .. DmatchFrameNWithFrameN_N]]}
# match[i] = list of all matches of frame i with all other frames
# match[i][j] = list of all matches of frame i with frame j