import cv2
import os
from schema import SOURCE_VIDEO_DIR, OUTOUT_FOLDER


for video_name in os.listdir(SOURCE_VIDEO_DIR):
    
    # loop thhrough all videos in raw_data folder 
    video = video_name.replace('.mp4', '')
    video_source = os.path.join(SOURCE_VIDEO_DIR, video)
    video_path = os.path.join(OUTOUT_FOLDER, video)
    original_frames_path = os.path.join(video_path, 'original_frames')
    final_frames = os.path.join(video_path, 'final_frames')        

    output_dir = os.path.join(final_frames, 'videos')


    # Directory containing frames
    frame_folder = final_frames

    # Sort the files in numerical order
    frame_files = sorted(os.listdir(frame_folder), key=lambda x: int(x.split('_')[0]))

    # Read the first frame to determine the size
    first_frame = cv2.imread(os.path.join(frame_folder, frame_files[0]))
    height, width, layers = first_frame.shape

    # Video properties
    fps = 25
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # or use 'MP4V' for .mp4 format

    # Create VideoWriter object
    video = cv2.VideoWriter(os.path.join(output_dir, video_name), 
                            fourcc, 
                            fps, 
                            (width, height))

    # Read each frame and write it to video
    for frame_file in frame_files:
        frame_path = os.path.join(frame_folder, frame_file)
        frame = cv2.imread(frame_path)
        video.write(frame)

    # Release the VideoWriter
    video.release()
