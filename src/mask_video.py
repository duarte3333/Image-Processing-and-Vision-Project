import cv2
import os
import numpy as np

def horizon_mask(video_path, video_name, fps=60, codec="MJPG"):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*codec)
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Create a blue rectangle mask for the upper third of the frame
        mask = frame.copy()
        mask[:height // 5, :, :] = [255, 0, 0]  # Blue color

        # Replace the original frame with the mask
        frame_with_mask = frame.copy()
        frame_with_mask[:height // 5, :, :] = mask[:height // 5, :, :]

        video.write(frame_with_mask)
        cv2.imshow('Video with Mask', frame_with_mask)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    video.release()

def bumper_mask(input_video, video_name, fps=60, codec="MJPG"):
    cap = cv2.VideoCapture(input_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Reduce frame size to 720p resolution
    new_width = 1280
    new_height = 720

    fourcc = cv2.VideoWriter_fourcc(*codec)
    video = cv2.VideoWriter(video_name, fourcc, fps, (new_width, new_height))

    # Calculate the center and axes of the ellipse
    center = (new_width // 2, new_height // 2)
    axes = (new_width // 2, new_height // 2)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to 720p resolution
        frame = cv2.resize(frame, (new_width, new_height))

        # Create a blue mask with the same size as the frame
        mask = np.zeros_like(frame)
        mask[:, :, 0] = 255  # Set blue channel to 255

        # Draw an ellipse on the mask
        cv2.ellipse(mask, center, axes, 0, 0, 360, (255, 255, 255), -1)

        # Apply the mask to the frame
        frame_with_mask = cv2.bitwise_and(frame, mask)

        video.write(frame_with_mask)
        cv2.imshow('BACK video with bumper mask', frame_with_mask)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    video.release()

def left_mask(video_path, video_name, fps=60, codec="MJPG"):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*codec)
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Create a blue rectangle mask for the left side of the frame
        mask = frame.copy()
        mask[:, :width // 10, :] = [255, 0, 0]  # Blue color

        # Add a small rectangle at the top left corner
        rect_width = width // 3
        rect_height = height // 9
        mask[:rect_height, :rect_width, :] = [255, 0, 0]  # Blue color

        # Replace the original frame with the mask
        frame_with_mask = frame.copy()
        frame_with_mask[:, :width, :] = mask[:, :width, :]

        video.write(frame_with_mask)
        cv2.imshow('LEFT video with mask', frame_with_mask)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    video.release()

def right_mask(video_path, video_name, fps=60, codec="MJPG"):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*codec)
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Create a blue rectangle mask for the right side of the frame
        mask = frame.copy()
        mask[:, -width // 10:, :] = [255, 0, 0]  # Blue color

        # Add a small rectangle at the top right corner
        rect_width = width // 3
        rect_height = height // 9
        mask[:rect_height, -rect_width:, :] = [255, 0, 0]  # Blue color

        # Replace the original frame with the mask
        frame_with_mask = frame.copy()
        frame_with_mask[:, :width, :] = mask[:, :width, :]

        video.write(frame_with_mask)
        cv2.imshow('RIGHT video with mask', frame_with_mask)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    video.release()

video_paths = ["video\TESLA_IST_ORIGINAL\TESLA_IST_ORIGINAL\\2023-04-29_16-40-01-back.mp4",
               "video\TESLA_IST_ORIGINAL\TESLA_IST_ORIGINAL\\2023-04-29_16-40-01-front.mp4",
               "video\TESLA_IST_ORIGINAL\TESLA_IST_ORIGINAL\\2023-04-29_16-40-01-left.mp4",
               "video\TESLA_IST_ORIGINAL\TESLA_IST_ORIGINAL\\2023-04-29_16-40-01-right.mp4"]

back_masked = bumper_mask(video_paths[0], "back_video_masked.avi")
# front_masked = front video does not need masking. 
left_masked = left_mask(video_paths[2], "left_video_masked.avi")
right_masked = right_mask(video_paths[3], "right_video_masked.avi")


"""
#  Example usage:
input_video = "video\TESLA_IST_ORIGINAL\TESLA_IST_ORIGINAL\\2023-04-29_16-40-01-left_repeater.mp4"
video_name = "example_video_with_mask.avi"
left_mask(input_video, video_name)
"""