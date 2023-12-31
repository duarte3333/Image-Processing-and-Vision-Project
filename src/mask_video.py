import cv2
import os
import numpy as np

def horizon_mask(image_folder, video_name, fps=30, codec="MJPG"):
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images.sort()  # Sort the images to ensure proper order

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*codec)
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for image in images:
        img_path = os.path.join(image_folder, image)
        img = cv2.imread(img_path)

        # Create a blue rectangle mask for the upper third of the image
        mask = img.copy()
        mask[:height // 5, :, :] = [255, 0, 0]  # Blue color

        # Replace the original frame with the mask
        img_with_mask = img.copy()
        img_with_mask[:height // 5, :, :] = mask[:height // 5, :, :]


        video.write(img_with_mask)
        cv2.imshow('Video with Mask', img_with_mask)
        cv2.waitKey(25)  # Adjust the delay between frames (in milliseconds)

    cv2.destroyAllWindows()
    video.release()

def bumper_mask(image_folder, video_name, fps=30, codec="MJPG"):
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images.sort()  # Sort the images to ensure proper order

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*codec)
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    # Calculate the center and axes of the ellipse
    center = (width // 2, height // 2)
    axes = (width // 2, height // 3)

    for image in images:
        img_path = os.path.join(image_folder, image)
        img = cv2.imread(img_path)

        # Create a blue mask with the same size as the image
        mask = np.zeros_like(img)
        mask[:, :, 0] = 255  # Set blue channel to 255

        # Draw an ellipse on the mask
        cv2.ellipse(mask, center, axes, 0, 0, 360, (255, 255, 255), -1)

        # Apply the mask to the image
        img_with_mask = cv2.bitwise_and(img, mask)

        video.write(img_with_mask)
        cv2.imshow('Video with Mask', img_with_mask)
        cv2.waitKey(25)  # Adjust the delay between frames (in milliseconds)

    cv2.destroyAllWindows()
    video.release()

# Example usage:
image_folder = "video/back"
video_name = "output_video_with_mask.avi"
bumper_mask(image_folder, video_name)




