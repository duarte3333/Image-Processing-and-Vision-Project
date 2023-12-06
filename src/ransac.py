import numpy as np
from numpy.linalg import eig
from src.homography import *

def RANSAC(src,dst,iter,threshold):
      best_homography = None
      inliers = 0
      for t in range(iter):
            sample_indices = np.random.choice(int(len(src)), size=4, replace=False)
            #int(len(src)*0.1)
            # Compute the Homography
            src_homography = [src[j] for j in sample_indices]
            dst_homography = [dst[j] for j in sample_indices]

            H = compute_homography(src_homography,dst_homography)
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
            if inl > inliers:
                 best_homography = H
                 inliers = inl
      return best_homography, inliers


