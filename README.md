# Image Processing and Vision

## Key concepts
### What is a homography?

A homography is a mathematical transformation that describes the relationship between two sets of points in different images, representing a planar geometric distortion or perspective transformation.

<p align="center">
  <img src="https://github.com/duarte3333/Image-Processing-and-Vision-Project/assets/76222459/2cb029ac-1628-41d0-acc5-f6c8cafaf6a7" alt="Image Description" style="width: 80%;">
</p>

The linear system:
<p align="center">
  <img src="https://github.com/duarte3333/Image-Processing-and-Vision-Project/assets/76222459/575bf6ce-d435-41ad-bbe1-f35b92f6fa47" style="width: 60%;">
</p>


### What is RANSAC?
RANSAC (Random Sample Consensus) is a robust iterative method used to estimate parameters of a mathematical model from a set of observed data 
points by iteratively selecting random subsets and fitting the model to each subset while identifying the consensus subset that best represents the underlying model.

<p align="center">
  <img src="https://github.com/duarte3333/Image-Processing-and-Vision-Project/assets/76222459/3aa567e6-5b6e-4dd3-976c-65d0e3629261" style="width: 50%;">
</p>


### What is a Pinhole Camera Model?
The pinhole camera model describes a simplified imaging system where light from a scene passes through a small aperture (pinhole) to form an inverted and reversed image on the image plane, with no lens distortion.

<p align="center">
  <img src="https://github.com/duarte3333/Image-Processing-and-Vision-Project/assets/76222459/9787dc74-5b64-4d73-987e-dfccfe62b8df" style="width: 60%;">
</p>

<p align="center">
  <img src="https://github.com/duarte3333/Image-Processing-and-Vision-Project/assets/76222459/b89fe5fa-90d4-4aac-80a2-66cda209bd3f" style="width: 60%;">
</p>

<p align="center">
  <img src="https://github.com/duarte3333/Image-Processing-and-Vision-Project/assets/76222459/827310a8-940b-4b2f-b579-dc6820bb569f" style="width: 40%;">
</p>

## Objectives
For **Part 1**, given a video and configuration file we either
compute: **All homographies**, that is, we compute the homographies
between all the frames of the given video. Or **Map homographies**,
where we compute the homographies between the frames of the video and
the given map, or initial frame. The main processing modules are
described in the following diagram:

<p align="center">
  <img src="https://github.com/duarte3333/Image-Processing-and-Vision-Project/assets/76222459/be25e536-cf68-4e0c-ab7f-0f383c0d4a20" alt="Image Description" style="width: 70%;">
</p>

**Figure 1:** Part 1 Main Processing Modules

For **Part 2** we used the *TESLA IST ORIGINAL* back camera video and
its camera intrinsic parameters to compute its Translation and
Rotation between certain frames of the video. The motivation for this
part is to setup the foundations to a program that can track the
trajectory of the car, based on a video from it's camera. The main
processing modules can be expressed in the following diagram:


<p align="center">
  <img src="https://github.com/duarte3333/Image-Processing-and-Vision-Project/assets/76222459/e189b261-3a0f-4d6a-9130-f22187b92eb9" alt="Image Description" style="width: 80%;">
</p>

**Figure 2:** Part 2 Main Processing Modules

## Methods

**2.1** **Part 1**

<p align="center">
  <img src="https://github.com/duarte3333/Image-Processing-and-Vision-Project/assets/76222459/1e0ae32d-793b-44c8-b287-f27d1d58ab2a" alt="Image Description" style="width: 80%;">
</p>

1\. **Feature Extraction (SIFT)**: In this module we extract the
frames from the video and convert them into grayscale. Afterwards we
extract the SIFT features, encoded as descriptors, from them.

2\. **Feature Matching (KNN)**: It starts by fitting the descriptors
of the first frame to a K-Nearest Neighbors model. Then, it finds its
two nearest neighbors in the descriptors of the second frame. Finally,
it applies a threshold to determine good matches and checks if there
are any 2 features that are matched to the same one (eliminating the
worst match).

3\. **RANSAC**: This classifier module works by selecting four random
matches and computing an homography from them, using SVD (Single Value
Decomposition). Subsequently the the best possible homography using LS (Least Squares).

4\. **Compute Final Homography**: To compute the homographies from
each frame to the map or from each frame to each other, we start by
defining the *sequential* homography from each frame to the next frame
in the video timeline (using RANSAC). After this, to compute the
homography between frames that are not in sequence, we start by
multiplying the *sequential* homographies with each other, starting
from one of the frames until we reach the destination frame. This
homography has some built up error but it enables us to check if one
frame overlapses with the other. If they do overlap, then a *direct*
homography is made from one frame to the other, in which case we also
do feature matching and RANSAC, (which carries less error than the
*sequential* one). This is repeated for all pairs of frames, but
instead of using all *sequential* homographies from one frame to the
other, some previously calculated *direct* homographies are used
alongside the necessary *sequential* homographies.

In order to check if the two frames overlap, the previously calculated
homography is used to calculate the coordinates of the source-frame's
corners in the destination frame. The previously calculated homography
is replaced by a *direct* one if all the transformed corners appear in
the new frame, unless these are entirely on the same vertical or
horizontal half-plane. This approach ensures computations of *direct*
homographies only for frames with significant overlap.

**2.2** **Part 2**

<p align="center">
  <img src="https://github.com/duarte3333/Image-Processing-and-Vision-Project/assets/76222459/a4aa20e2-0779-49be-8387-acd0c09f763a" alt="Image Description" style="width: 80%;">
</p>

1\. **Video Masking**: Due to the fish-eye camera, videos include
parts of the car which can be captured as features, hence making our
task troublesome. For the rear camera video, the bumper and license
plate are masked with an elliptical mask, (essentially a blue color
block), due to the radial distortion.

2\. **Feature Extraction (SIFT)**: Likewise before, we convert the
frames from the masked video into grayscale and then extract the SIFT
features, encoded as descriptors.

3\. **Feature Matching (KNN)**: This is the same function as described
in Part 1.

4\. **Essential Matrix Computation**: Having the matches, we
*undistort* the points us-ing the Radial Distortion components of the
back camera. Next we find the Essential matrix using
*cv2.findEssentialMat*, with *cv2.RANSAC*. This takes into
consideration the intrinsics of the camera, which are given for this
part.

5\. **Rotation and Translation Computation**: Finally, we recover the
most probable Ro-tation and Translation of the back camera
(considering the matched points), between two certain frames, by
employing *cv2.recoverPose*.

So as to debug this part, we started by using features selected and
matched by hand (in order for the matched points to not be error
inducing). We used 8 features spread out through the image (more than
the 5 unknowns of the Essential matrix). After this, we tested if the
Epipolar Lines that were generated by the Essential matrix made sense
(i.e if they all intersect in one point within each frame). Lastly,
the Rotation and Translation were computed, and these were checked to
make sure that there was almost no rotation and only translation in
one axis, between frames.
