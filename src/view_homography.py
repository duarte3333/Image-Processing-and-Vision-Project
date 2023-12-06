import cv2
import numpy as np
from src.matching_features import matching_features, matching_features_matrix
from src.ransac import RANSAC
from src.homography import create_homography_openCV, create_src_dest, test_homography

def warp_and_display_frame(frame_N, H_array, frame_dict):
    """
    Warp and display the first frame as seen from the perspective of frame N
    """
    
    height, width = frame_N.shape[:2]
    warped_first_frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Warp the first frame using the homography matrices
    for i, H_ij in homography_matrices.items():
        if i in frame_dict:
            warped_frame_i = cv2.warpPerspective(frame_dict[i], H_ij, (width, height))
            warped_first_frame += warped_frame_i

    # Display the original frame N and the warped first frame
    cv2.imshow('Original Frame N', frame_N)
    cv2.imshow('Warped First Frame as Seen from N', warped_first_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return warped_first_frame

def view_homography_two_frames(img1, img2, sift_points, kp_list, option):
    """ Compute the homography for two frames """
    match = matching_features(sift_points, img1, img2, cv2.SIFT_create(5000))
    print("Number of matches: ", len(match), '\n')
    if (option == 1):
        src, dst, H  = create_homography_openCV(match, kp_list, 0, 19)
    if (option == 2):
        print(match)
        src1, dst1 = create_src_dest(match, kp_list, 0, 19)
        print("src: ", len(src1), '\n')
        print("dst: ", dst1, '\n')
        H, inliers = RANSAC(src1, dst1, 50, 0.8)
    test_homography(img1, img2, H)