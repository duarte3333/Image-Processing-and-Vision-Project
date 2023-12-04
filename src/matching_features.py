import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors

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

def matching_features_SCIKITLEARN(sift_points):
    """Feature matching using nearest neighbours, for pairs of consecutive frames"""
    
    matches=[]
    Threshold=0.75

    for s in range(len(sift_points)-1):
        frame1_descriptors = sift_points[s][2:,:] #descriptor values of every feature point for video frame s (current shape: 128x5000)
        frame1_descriptors = np.transpose(frame1_descriptors) # transpose -> current shape: 5000x128 - > 5000 points/queries each with 128 features/columns
        #fit data of features from frame 1 to NearestNeighbour. When we ask for matches from this method, it should give us the 2 closest points to the point given
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(frame1_descriptors) 

        #predict matches for the other frame:
        
        frame_drescriptors = np.transpose(sift_points[s+1][2:,:]) #the same as done some lines above but for frame s+1
        # Find the 2 nearest neighbors
        distances, indices = nbrs.kneighbors(frame_drescriptors) 
        # indices is a 5000x2 shape matrix -> for each of the 5000 given feature points of frame_drescriptors it gives the 2 closest features from video frame 1
        # distances is a measure of distance between the feature points of frame_drescriptors and each of the two givenneighbours from the indices matrix - it has the same size as indices
        
        features_matches=np.empty([4,0])
        features_not_mateched=[]
        for i in range(len(distances)): #loop (number of features from sift)
            if distances[i,0]< Threshold*distances[i,1] and distances[i,0]< 700:
                #match is good for first neighbour found
                features_matches= np.hstack((  features_matches   , np.array([[int(i)],[int(indices[i,0])], [distances[i,0]],[distances[i,1]]])  ))
            else:                
                #point is not good
                features_not_mateched.append(i) #features from this frame that were not matched
        
        features_matches = features_matches[:, features_matches[1, :].argsort()] # this sorts the check_for_duplicates matrix in accordance to the values of it's second line
        features_matches_deletedColumns= features_matches.copy()

        for i in reversed (range (1, features_matches.shape[1])): #loop that starts in the last feature - because it deletes elements with their indexes from list check_for_duplicates_deletedColumns
            # this has to be done starting from the end to not change the index of columns

            # duplicates are adjacent because of sort
            if features_matches[1,i-1] == features_matches[1,i]:
                # if the value of the indice i and i-1 are equal, then there is one feature matched to 2 features of the new frame - we need to delete one of the matches
                if features_matches[2,i-1] <= features_matches[2,i]: #check distance of i and i-1. And remove the one with the most distance
                    features_matches_deletedColumns= np.delete(features_matches_deletedColumns, i-1, 1) #remove duplicate feature matching (deletes one column - np dimension 1)
                    features_not_mateched.append(features_matches[0,i-1]) #append number of feature that was deleted to features not matched
                else:
                    features_matches_deletedColumns= np.delete(features_matches_deletedColumns, i, 1) 
                    features_not_mateched.append(features_matches[0,i]) 
        
        matched_inThis_frame = features_matches_deletedColumns[:, features_matches_deletedColumns[0, :].argsort()] #to be in order in acoordance to index of frame s

        matches.append( (matched_inThis_frame[0:2,:]))

    return matches

def matching_features_matrix(sift_1frame, sift_otherframes): 
    """This matches the features of sift_1frame (one frame) to the features of each frame of sift_otherframes
    It returns a list of arrays. Each array is the matching between the first frame and one other frame. 
    This matching is an array with 2 lines with indexes corresponding to the first frame in the first line and the other frame in the second line. Indexes in the same column are matches"""
    frame1_descriptors = sift_1frame[2:,:] #descriptor values of every feature point for video frame 1 (current shape: 128x5000)
    frame1_descriptors = np.transpose(frame1_descriptors) # transpose -> current shape: 5000x128 - > 5000 points/queries each with 128 features/columns


    #fit data of features from frame 1 to NearestNeighbour. When we ask for matches from this method, it should give us the 2 closest points to the point given
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(frame1_descriptors) 

    matches=[]
    #predict matches for other frames -> sift_points[1:] does not count with first frame
    Threshold=2

    for i in range(len(sift_otherframes)):
        frame_drescriptors = np.transpose(sift_otherframes[i][2:,:]) #the same as done some lines above but for frame i
        # Find the 2 nearest neighbors
        distances, indices = nbrs.kneighbors(frame_drescriptors) 
        # indices is a 5000x2 shape matrix -> for each of the 5000 given feature points of frame_drescriptors it gives the 2 closest features from video frame 1
        # distances is a measure of distance between the feature points of frame_drescriptors and each of the two givenneighbours from the indices matrix - it has the same size as indices
        features_matches=np.empty([4,0])
        features_not_mateched=[]
        for k in range(len(distances)):
            if distances[k,0]> Threshold:
                #point is not good
                features_not_mateched.append(k) #features from this frame that were not matched
            else:
                #match is good for first neighbour found
                features_matches= np.hstack((  features_matches   , np.array([[int(k)],[int(indices[k,0])], [distances[k,0]],[distances[k,1]]])  ))


        features_matches = features_matches[:, features_matches[1, :].argsort()] # this sorts the check_for_duplicates matrix in accordance to the values of it's second line
        features_matches_deletedColumns= features_matches.copy()

        for i in reversed (range (1, features_matches.shape[1])): #loop that starts in the last feature - because it deletes elements with their indexes from list features_matches_deletedColumns
            # this has to be done starting from the end to not change the index of columns

            # duplicates are adjacent because of sort
            if features_matches[1,i-1] == features_matches[1,i]:
                # if the value of the indice i and i-1 are equal, then there is one feature matched to 2 features of the new frame - we need to delete one of the matches
                if features_matches[2,i-1] <= features_matches[2,i]: #check distance of i and i-1. And remove the one with the most distance
                    features_matches_deletedColumns= np.delete(features_matches_deletedColumns, i-1, 1) #remove duplicate feature matching (deletes one column - np dimension 1)
                    features_not_mateched.append(features_matches[0,i-1]) #append number of feature that was deleted to features not matched
                else:
                    features_matches_deletedColumns= np.delete(features_matches_deletedColumns, i, 1) 
                    features_not_mateched.append(features_matches[0,i]) 
        
        matched_inThis_frame = features_matches_deletedColumns[:, features_matches_deletedColumns[0, :].argsort()]
        matches.append(matched_inThis_frame[0:2,:])
    return matches