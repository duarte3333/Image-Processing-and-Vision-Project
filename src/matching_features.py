import cv2
import numpy as np

def matching_features(sift_points):
    #Brute force method
    bf = cv2.BFMatcher(crossCheck=True) #crossCheck is set to true so that the match is symmetric
    all_matches = []
    match = []
    for s in range(len(sift_points)-1):
        point_matches = []
        des1 = (((sift_points[s])[2:,:])).astype('float32')  # descriptors of the first frame
        des2 = (((sift_points[s+1])[2:,:])).astype('float32')  # descriptors of the second
        des1 = np.reshape(des1,(np.shape(des1)[1],128))
        des2 = np.reshape(des2,(np.shape(des2)[1],128))
        if np.shape(des1)[0] > np.shape(des2)[0]:
                 des1 = des1[:-abs(np.shape(des1)[0]-np.shape(des2)[0]),:]  # we are removing the last points so that we have an equal amount of SIFT features between two frames
        if np.shape(des1)[0] < np.shape(des2)[0]:
                 des2 = des2[:-abs(np.shape(des1)[0]-np.shape(des2)[0]),:]
        matches = bf.match(des1,des2)  # an error occurs if two frames have different amounts of SIFT features
        # try:
        # except: 
        #       print('Error!')        
        for i in range(len(matches)):
            match.append(matches)
            point_matches.append([matches[i].queryIdx,matches[i].trainIdx])
        all_matches.append(point_matches)
    return match