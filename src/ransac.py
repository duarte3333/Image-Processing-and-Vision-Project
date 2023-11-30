import numpy as np
from numpy.linalg import eig
from src.homography import *

def RANSAC(src,dst,iter,threshold, eigenValues):
      best_homography = None
      inliers = [0]
      for t in range(iter):
            sample_indices = np.random.choice(int(len(src)), size=8, replace=False)
            #int(len(src)*0.1)
            # Compute the Homography
            H = compute_homography(src[sample_indices],dst[sample_indices], eigenValues)
            inl = 0
            for p, q in zip(src, dst):
                x1 = p[0]
                y1 = p[1]
                x2 = q[0]
                y2 = q[1]
                # Transform the point using the estimated homography
                transformed_point = np.dot(H, np.array([x1, y1, 1]))
                # Normalize the transformed point
                transformed_point /= transformed_point[2]
                # Calculate the Euclidean distance between the transformed point and the actual point
                distance = np.linalg.norm(np.array([x2, y2, 1]) - transformed_point)
                if distance < threshold:
                   inl += 1
            if inl > inliers[0]:
                 best_homography = H
                 inliers[0] = inl
      return best_homography, inliers[0] 