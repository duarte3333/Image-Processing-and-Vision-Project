import cv2
import numpy as np

def matching_features(sift_points, img1, img2, sift):
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # BFMatcher with default params
    bf = cv2.BFMatcher() 
    matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
    all_matches = []
    good_matches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good_matches.append([m])

    # cv.drawMatchesKnn expects list of lists as matches.
    #img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #plt.imshow(img3),plt.show()
    return good_matches