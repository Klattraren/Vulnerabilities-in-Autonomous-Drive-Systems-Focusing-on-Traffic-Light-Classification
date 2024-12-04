import cv2
import os
from detection_tmp import detect_traffic_lights

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
    detect_traffic_lights(temp_dir, model_name, Num_images=num_images, plot_flag=True, padding=2)



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

# # Usage Example
# input_video_path = './working_model_clean_data/video_processing/IMG_5189.mp4'
# output_video_path = './working_model_clean_data/video_processing/output_video.mp4'
# temp_dir = "./working_model_clean_data/video_processing/temp_frames"
# temp_after = "./working_model_clean_data/video_processing/temp_after"

# # Split video into frames and run traffic light detection
# MODEL_NAME = 'faster_rcnn_resnet101_coco_11_06_2017'  # for improved accuracy
# Num_images = len([name for name in os.listdir(temp_dir) if os.path.isfile(os.path.join(temp_dir, name))])
# split_video_into_frames(input_video_path, temp_dir, MODEL_NAME, Num_images)

# # Get video properties
# video = cv2.VideoCapture(input_video_path)
# fps = video.get(cv2.CAP_PROP_FPS)
# frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
# video.release()

# # Process frames and create video
# process_frames_and_create_video(temp_after, output_video_path, fps, frame_width, frame_height)
