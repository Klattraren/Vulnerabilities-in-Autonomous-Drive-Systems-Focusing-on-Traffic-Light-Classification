import cv2
import os
from .custom_detection import detect_traffic_lights

def split_video_into_frames(input_video_path, temp_dir, model_name, num_images):
    """
    Splits a video into frames, saves them to a temporary directory, and runs detect_traffic_lights.

    Args:
        input_video_path (str): Path to the input MP4 video.
        temp_dir (str): Path to the temporary directory to save frames.
        model_name (str): Name of the model to use for traffic light detection.
        num_images (int): Number of images to detect.
    """
    os.makedirs(temp_dir, exist_ok=True)

    # Open the video file
    video = cv2.VideoCapture(input_video_path)
    frame_idx = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        # Save the frame to the temp directory
        frame_path = f"frame_{frame_idx:04d}.png"
        cv2.imwrite(os.path.join(temp_dir, frame_path), frame)
        frame_idx += 1
    

    video.release()
    print(f"Frames saved to: {temp_dir}")

    # Run detect_traffic_lights on the saved frames
    # Returns detected boxes, classes and scores.
    box_list, class_list, score_list = detect_traffic_lights(temp_dir, model_name, Num_images=num_images, padding=2)
    print(f"Detected {len(box_list)} traffic lights")
    return box_list, class_list, score_list



def process_frames_and_create_video(temp_after, output_video_path, fps, frame_width, frame_height):
    """
    Creates a video from the processed frames.

    Args:
        temp_after (str): Path to the temporary directory for processed frames.
        output_video_path (str): Path to save the processed video.
        fps (float): Frame rate for the output video.
        frame_width (int): Width of the video frames.
        frame_height (int): Height of the video frames.
    """
    os.makedirs(temp_after, exist_ok=True)

    # Create a video writer for the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Write the processed frames to the output video
    for frame_file in sorted(os.listdir(temp_after)):
        frame_path = os.path.join(temp_after, frame_file)
        frame = cv2.imread(frame_path)
        output_video.write(frame)

    output_video.release()

    print(f"Processed video saved to: {output_video_path}")
