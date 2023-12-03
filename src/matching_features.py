import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors


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

def matching_features_SCIKITLEARN(sift_points):
    """Feature matching using nearest neighbours, for pairs of consecutive frames"""
    
    matches=[]
    Threshold=2

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
            if distances[i,0]> Threshold:
                #point is not good
                features_not_mateched.append(i) #features from this frame that were not matched
            else:
                #match is good for first neighbour found
                features_matches= np.hstack((  features_matches   , np.array([[int(i)],[int(indices[i,0])], [distances[i,0]],[distances[i,1]]])  ))


        
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
        
        matches.append( (matched_inThis_frame[0:2,:]).tolist())

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