import os
from moviepy.editor import VideoFileClip
from schema import SOURCE_VIDEO_DIR

# Set your source and destination folder paths
source_folder = 'videos_from_the_sun_yt/'

# Ensure the destination folder exists
os.makedirs(SOURCE_VIDEO_DIR, exist_ok=True)
counter = 0

# Default frame rate (you can adjust this as needed)
default_fps = 30

# Loop through each file in the source folder
for filename in os.listdir(source_folder):
    if filename.endswith('.mp4'):
        # Construct the full path to the source video
        source_filepath = os.path.join(source_folder, filename)

        # Load the video
        video = VideoFileClip(source_filepath)

        # Use the video's fps if available, otherwise use the default
        video_fps = video.fps if video.fps is not None else default_fps
        print(video_fps)
        # Cut a 15-second segment starting from the 10th second
        edited_video = video.subclip(10, 25)  # From 10th to 25th second
        
        # Construct the path for the edited video
        destination_filepath = os.path.join(SOURCE_VIDEO_DIR, filename)

        # Write the edited video to the destination folder without audio
        edited_video.write_videofile(destination_filepath, codec="libx264", fps=video_fps, audio=False)
        counter += 1
        print(f'Video {counter} done!')

print("Video processing complete.")


