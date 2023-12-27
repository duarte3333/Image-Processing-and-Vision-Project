import cv2
import os

""" def frames_to_video(image_folder, video_name, fps=30, codec="MJPG"):
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images.sort()  # Sort the images to ensure proper order

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*codec)
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for image in images:
        img_path = os.path.join(image_folder, image)
        img = cv2.imread(img_path)
        video.write(img)
        cv2.imshow('Video', img)
        cv2.waitKey(25)  # Adjust the delay between frames (in milliseconds)

    cv2.destroyAllWindows()
    video.release()

# Example usage:
image_folder = "video/back"
video_name = "output_video.avi"
frames_to_video(image_folder, video_name) """



def frames_to_video_with_mask(image_folder, video_name, fps=30, codec="MJPG"):
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

        # Combine the original frame and the mask
        img_with_mask = cv2.addWeighted(img, 1, mask, .99, 0)

        video.write(img_with_mask)
        cv2.imshow('Video with Mask', img_with_mask)
        cv2.waitKey(25)  # Adjust the delay between frames (in milliseconds)

    cv2.destroyAllWindows()
    video.release()

# Example usage:
image_folder = "video/back"
video_name = "output_video_with_mask.avi"
frames_to_video_with_mask(image_folder, video_name)
