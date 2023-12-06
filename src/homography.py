from sklearn import preprocessing
import numpy as np
from numpy.linalg import eig
import cv2

def compute_homography(src, dst):
    """ This function computes the homography matrix 
    between two sets of points"""
    # src = preprocessing.normalize(src)   #Normalization
    A = []
    for p, q in zip(src, dst):
        x1 = p[0]
        y1 = p[1]
        x2 = q[0]
        y2 = q[1]
        A.append([-x1, -y1, -1, 0, 0, 0, x2*x1, x2*y1, x2])
        A.append([0, 0, 0, -x1, -y1, -1, y2*x1, y2*y1, y2])
    _, _, Vt = np.linalg.svd(A, full_matrices=True)
    x = Vt[-1]
    H = x.reshape(3, -1) / x[0]
    return H

def create_homography_openCV(match, kp_list, i, j):
    """ This function computes the homography matrix between two sets of points with cv2 """
    kp1 = kp_list[i]
    kp2 = kp_list[j]
    src_pts = np.float32([kp1[q[0].queryIdx].pt for q in match ]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[t[0].trainIdx].pt for t in match ]).reshape(-1,1,2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    print("Condition of H from cv2", np.linalg.cond(H))
    return src_pts, dst_pts, H

def create_src_dest(match, kp_list, i, j):
    kp1 = kp_list[i]
    kp2 = kp_list[j]
    #Reshaping the points so that they can be normalized
    src_pts = np.float32([ kp1[q.queryIdx].pt for q in match[0] ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[t.trainIdx].pt for t in match[1] ]).reshape(-1,1,2)
    src = np.reshape(src_pts,(np.shape(src_pts)[0],2))
    dst = np.reshape(dst_pts,(np.shape(dst_pts)[0],2))
    return src, dst

def apply_homography(image, H):
    """ Apply homography to the image """
    warped_img = cv2.warpPerspective(image, H, (image.shape[1], image.shape[0]))
    return warped_img

def test_homography(img_src, img_dest, homography):
    warped_src = apply_homography(img_src, homography)
    #print('condition:',np.linalg.cond(H),'inliers: ', inliers)
    cv2.imshow('Warped Source Image', warped_src)
    cv2.imshow('Source Image', img_src)
    cv2.imshow('Destination Image', img_dest)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
   
##COMMENTS 

# The variable good_matches is a list of lists of 
# DMatch objects. Each DMatch object contains 
# the following attributes: distance, imgIdx, 

# match = [q[0]=Dmatch1, ... , DmatchN] (size of the ones that match- 100)
# DmatchN = [distance, imgIdx, queryIdx, trainIdx]
# src_pts = [kp1[Dmatch1[0].queryIdx].pt, ... , Dmatch1[n].queryIdx].pt] (size of the ones that match- 100)
# kp1[Dmatch1[0].queryIdx].pt = (x,y) of the point in the image
#[[idx_i][idx_j]]

# match[j]=[[1,2,3,4,5,6,7,8],[36478,574382,54324,567,567,45,12]]

# eigenvalue,eigenvector=eig(np.matmul(np.transpose(A),A)) 
# idx = eigenvalue.argsort() #sorts the eigenvalues in ascending order
# eigenValues = eigenvalue[idx] #eigenvalues in ascending order
# eigenVectors = eigenvector[:,idx] #eigenVectors in ascending order
# x=eigenVectors[0]

# Condition -> A*A^(-1) - High condition means small changes in the input can result in large changes in the output

# H_output
## indexframedst
## indexframesrc
## H11
## H12
## H13
## H21           
## H22
## H23
## H31
## H32
## H33  