from sklearn import preprocessing
import numpy as np
from numpy.linalg import eig

def compute_homography(src, dst, eigenValues):
        A = []
        for p, q in zip(src, dst):
            x1 = p[0]
            y1 = p[1]
            x2 = q[0]
            y2 = q[1]
            A.append([-x1, -y1, -1, 0, 0, 0, x2*x1, x2*y1, x2])
            A.append([0, 0, 0, -x1, -y1, -1, y2*x1, y2*y1, y2])

        eigenvalue,eigenvector=eig(np.matmul(np.transpose(A),A)) 
        idx = eigenvalue.argsort() #sorts the eigenvalues in ascending order
        # print("idx", idx)
        # print("eigenValues", eigenValues)
        eigenValues = eigenvalue[idx] #eigenvalues in ascending order
        eigenVectors = eigenvector[:,idx] #eigenVectors in ascending order
        # print("eigenVectors", eigenVectors)
        # print("eigenValues", eigenValues[-1])
        #_, _, Vt = np.linalg.svd(A, full_matrices=True)
        #x = Vt[-1]
        x=eigenVectors[0]
        H = x.reshape(3, -1) / x[0]
        return H


def create_homography(match, kp_list):
    kp1 = kp_list[0]
    kp2 = kp_list[1]
    
    #Reshaping the points so that they can be normalized
    src_pts = np.float32([ kp1[q.queryIdx].pt for q in match[0] ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[t.trainIdx].pt for t in match[1] ]).reshape(-1,1,2)
    src = np.reshape(src_pts,(np.shape(src_pts)[0],2))
    dst = np.reshape(dst_pts,(np.shape(dst_pts)[0],2))
    src = preprocessing.normalize(src)   #Normalization
    dst = preprocessing.normalize(dst)

    A = []
    for p, q in zip(src, dst):
                x1 = p[0]
                y1 = p[1]
                x2 = q[0]
                y2 = q[1]
                A.append([-x1, -y1, -1, 0, 0, 0, x2*x1, x2*y1, x2])
                A.append([0, 0, 0, -x1, -y1, -1, y2*x1, y2*y1, y2])
    _, _, Vt = np.linalg.svd(A, full_matrices=True)

    eigenvalue,eigenvector=eig(np.matmul(np.transpose(A),A))

    H = np.reshape(eigenvector[0],(3,3))
    print(np.linalg.cond(H))

    H2 =  Vt[-1,:].reshape(3, 3)
    print(np.linalg.cond(H2))
    idx = eigenvalue.argsort()[::-1]   
    eigenValues = eigenvalue[idx]
    eigenVectors = eigenvector[:,idx]
    #eigenValues[-1]
    return src, dst, eigenValues
