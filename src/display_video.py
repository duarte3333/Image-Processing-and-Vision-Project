import cv2
import os

def display_video(video_path):
    """ Display the video """
    capture = cv2.VideoCapture(os.path.abspath(video_path))
    if not capture.isOpened():
        print("Error: Unable to open video file.")
        return
    while True:
        ret, frame = capture.read()
        if not ret:
            print("Error: Failed to read frame.")
            break
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()