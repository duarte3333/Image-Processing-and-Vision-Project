import numpy as np
from numpy.linalg import eig
import cv2
from src.ransac import *
from src.matching_features import *


def recalculate_homographies_if_intersection(H_output, height, width, sift_points, matches):    
    # Iterate over each homography in H_output
    for i in range(1, H_output.shape[1]):
        
        H = H_output[2:, i].reshape((3, 3)) # Extract the homography matrix H from the current column of H_output
        index_frame_i = int(H_output[0, i] - 1) # Index of frame i in sift_points
        index_frame_j = int(H_output[1, i] - 1) # Index of frame j in sift_points
        
        #Coordinates of the corners
        transformed_corners = calculate_transformed_corners(H, width, height) # corners coordinates in destination frame (frame 8)
        original_corners = find_original_coordinates(np.linalg.inv(H), transformed_corners) # corners coordinates in origin frame (frame 1)
        
        #Features of frames
        features_frame_i = sift_points[index_frame_i]  # Features from frame 1 (origin)
        features_frame_j = sift_points[index_frame_j]  # Features from frame 8 (destination)
        
        # Filter features inside the corners for frame 1 and frame i
        filtered_features_frame_i = filter_features_outside_corners(features_frame_i, original_corners)
        filtered_features_frame_j = filter_features_outside_corners(features_frame_j, transformed_corners)
        
        # Check if all points are on one side of the centerline horizontally or vertically
        center_x, center_y = width // 2, height // 2
        all_above = all(c[1] < center_y for c in transformed_corners)
        all_below = all(c[1] > center_y for c in transformed_corners)
        all_right = all(c[0] > center_x for c in transformed_corners)
        all_left = all(c[0] < center_x for c in transformed_corners)
        

        # Recompute homography with filtered features
        if not (all_above or all_below or all_right or all_left):
            src_pts = filtered_features_frame_i  # List of (x, y) tuples
            dst_pts = filtered_features_frame_j  # List of (x, y) tuples
            
            new_H, _ = RANSAC(src_pts, dst_pts, 72, 0.8)
            
            # Replace the old homography with the new one in H_output
            H_output[2:, i] = new_H.reshape(9)
    
    return H_output

def calculate_transformed_corners(H, width, height):
    # Corners of the image in homogenous coordinates
    corners = np.array([
        [0, 0, 1], # Top left
        [width - 1, 0, 1], # Top right
        [width - 1, height - 1, 1], # Bottom right
        [0, height - 1, 1] # Bottom left
    ]).T
    
    # Apply the homography to the corners
    transformed_corners = H @ corners #this 
    transformed_corners /= transformed_corners[2] # Normalize the points
    print("coners", transformed_corners[:2].T)
    return transformed_corners[:2].T

def find_original_coordinates(H_inv, transformed_corners):    
    # Apply the inverse homography to the transformed corners
    original_corners = H_inv @ np.vstack([transformed_corners.T, np.ones((1, transformed_corners.shape[0]))])
    original_corners /= original_corners[2] # Normalize the points
    return original_corners[:2].T

def is_point_inside_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

def filter_features_outside_corners(keypoints, corners):
    # keypoints is a numpy array where each row is a keypoint and the first two columns are x and y coordinates.
    inside_keypoints = []
    for kp in keypoints:
        x, y = kp[0], kp[1]  # Extract the (x, y) coordinates from the keypoint
        if is_point_inside_polygon((x, y), corners):
            inside_keypoints.append((x, y))

    return inside_keypoints
