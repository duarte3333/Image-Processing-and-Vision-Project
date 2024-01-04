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
from src.display_video import *
from src.view_homography import *
from src.outputs import *
from src.optimized_corners import *
from scipy.io import loadmat


# OVERVIEW:
#   Feature detection: opencv
#   Matching : sklearn , numpy
#   Create Homography: numpy
#   RANSAC: numpy


def all_homographies(H_sequential,height, width, sift_points):
    """ This function computes the homography from any frame i to any frame j (Hij), with j>i
    This means that we are doing homography in the direction of a bigger frame, which is contrary to the direction of the H_sequential homographies
    To solve this, the H_sequential matrixes must be inverted before being used"""

    H_output=np.empty([11,0])
    
    for i in range(1,H_sequential.shape[1]+1): #H_sequential[0] is H_21
        for j in range(i+1, H_sequential.shape[1] +2):
            #the homographie between frame j-1 and j is:

            H_jminus1_j = np.linalg.inv(H_sequential[2:,j-2].reshape((3,3))) #H_sequential is H from 2 to 1, from 3 to 2, from 4 to 3... and we want the inverse direction
            
            if i+1 == j: #simple homographie
                T_to_map=  H_jminus1_j
            else: #compound of sequential homographies
                T_to_map_temporary= np.matmul( H_jminus1_j , H_output[2:,-1].reshape(3,3) ) # example: frame4 = H34*H23*H12*frame1
                T_to_map = recalculate_1_homography_if_intersection(T_to_map_temporary,height, width, sift_points, index_frame_dst=j-1, index_frame_src=i-1)

            H_i = np.vstack((np.array([[j],[i]] ), T_to_map.reshape(9,1) ))
            H_output = np.hstack([H_output, H_i])
        
    return H_output

def homography_to_map(H_sequential, H_frame1_to_map,height, width, sift_points):
    """ This function computes the homography from any frame to the map.
        For frame n, H_output[2:,i-1] should be the homography from frame n-1 to the map. 
        H_sequential[2:,i-1] should be the homography from frame n to n-1
        So T_to_Map should be the homography from frame n to map"""
    H_output=np.empty([11,0])
    H_frame_map=np.empty([11,0])
    H_i = np.vstack((np.array([[1],[2]] ), H_sequential[2:,0].reshape(9,1) )) 
    H_output = H_output = np.hstack([H_output, H_i])
    for i in range(1, H_sequential.shape[1]): #this should create H_output that translates from frame n to frame 1
        T_to_1= np.matmul(H_output[2:,i-1].reshape(3,3), H_sequential[2:,i].reshape(3,3))  
                                                                             #frame 1 in sift points is frame index 0
        T_to_1_direct=recalculate_1_homography_if_intersection(T_to_1,height, width, sift_points, index_frame_dst=0, index_frame_src=H_sequential[1,i]-1) #especificar matches e sif_points
        H_i = np.vstack((np.array([[1],[H_sequential[1,i]]] ), T_to_1_direct.reshape(9,1) )) #indices CHECK
        H_output = np.hstack([H_output, H_i])


    H_0 = np.vstack((np.array([[0], [1]]) , H_frame1_to_map.reshape(9,1) )) #first part of the array is 0 and 1 - which means homography from frame 1 to map (frame 0)
    H_frame_map = np.hstack([H_frame_map,H_0])
    for i in range(H_output.shape[1]): #translate homographjy from frame n to frame 1, into homography from frame n to frame 0(map)
        T_0=np.matmul(H_frame1_to_map,H_output[2:,i].reshape(3,3) )
        H_0 = np.vstack((np.array([[0], [ H_output[1,i] ]]) , T_0.reshape(9,1) ))
        H_frame_map=np.hstack([H_frame_map,H_0])
    return H_frame_map

def create_sequential_homographies(matches, sift_points):
    """ This function creates the homographies from frame n+1 to frame n.
        We want the homography from frame n+1 to frame n because we will 
        want the homography from any frame back to the frame 1 and then to the map."""
    H_sequential=np.empty([11,0])
    for i in range(len(matches)):
        kp_dst = sift_points[i][:2,:] #keyupoints for frame n (destination)
        kp_src= sift_points[i+1][:2,:] #keypoints for frame n+1 (origin)
        src_pts = [] #[[x1,y1, d1, ..., dn], [x1,y1, d1, ..., dn], ...]
        dst_pts = []
        for k in matches[i][0,:]:
            src_pts.append(  (float(kp_src[0,int(k)])   , float(kp_src[1,int(k)]))   )
        for k in matches[i][1,:]:
            dst_pts.append(  (float(kp_dst[0,int(k)])   , float(kp_dst[1,int(k)]))   )

        H_parameters, inliers = RANSAC(src_pts, dst_pts, 72, 0.8) #71.36 iterations to 0.99 suceess-> for 50% inliers
        indexes_frames = np.array([[i+1], [i+2]])
        H = np.vstack((indexes_frames, H_parameters.reshape(9,1) ))
        H_sequential = np.hstack([H_sequential, H])
        
    return H_sequential

def final_parsing_compute_transform(config_data):
    video_path = config_data[0].split(' ')[1].strip() #Get the video path
    type_homography = config_data[4].split(' ')[3] #Get the type of homography
    file_name_keypoints = "outputs/" + config_data[3].split(' ')[1] #Get the name of the keypoints file
    file_name_tranformations = "outputs/" + config_data[5].split(' ')[1] #Get the name of the transformations file
    width, height = config_data[6].split(' ')[1], config_data[6].split(' ')[2] #Get the width and height of the video
    return video_path, type_homography, file_name_keypoints, file_name_tranformations, width, height
    
def extract_keypoints(file_path):
    data = loadmat(file_path)
    keypoint_data = data['Keypoints']
    kp_list = []
    nr_points = None

    for item in keypoint_data:
        points = item.reshape((item.shape[1],item.shape[0])) # Transpose to revert the reshape operation
        kp_list.append(points)
        # Store the number of points (assuming all arrays have the same size)
        nr_points = points.shape[1] if nr_points is None else nr_points
    return kp_list, nr_points

if __name__ == "__main__":
    if (check_syntax()):
        sys.exit(1)
    config_data = parse_configuration_file(sys.argv[1]) #Parse the configuration file
    match_img1 , match_map = parse_points(config_data) #Parse the points from the configuration file
    video_path, type_homography, file_name_keypoints, file_name_tranformations, \
        width, height = final_parsing_compute_transform(config_data)
    
    H_frame1_to_map =compute_homography(match_img1, match_map)    
    #sift_points, nr_points = extract_features(video_path)
    sift_points, nr_points = extract_keypoints(file_name_keypoints)
    print("width, height", width, height)
    match = matching_features_SCIKITLEARN(sift_points)
    H_sequential = create_sequential_homographies(match, sift_points)
    if type_homography =='map':
        H_output = homography_to_map(H_sequential, H_frame1_to_map,height, width, sift_points)
    elif type_homography =='all':
        H_output = all_homographies(H_sequential,height, width, sift_points)
    print('H_output', H_output)
    
    #create_output_keypoints(sift_points, file_name_keypoints, nr_points)
    create_output(H_output, file_name_tranformations)        
    
