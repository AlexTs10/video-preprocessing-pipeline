import os
import cv2
from schema import SOURCE_VIDEO_DIR, OUTOUT_FOLDER

for video_name in os.listdir(SOURCE_VIDEO_DIR):

    # loop through all videos in raw_data folder 
    video = video_name.replace('.mp4', '')
    print(video)
    video_source = os.path.join(SOURCE_VIDEO_DIR, video_name)
    print(video_source)

    video_path = os.path.join(OUTOUT_FOLDER, video)
    os.makedirs(video_path, exist_ok=True)

    # Load the video
    cap = cv2.VideoCapture(video_source)
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    counter = 0
    # Read and process each frame
    while True:
        ret, frame = cap.read()

        # Break the loop if no frame is returned
        if not ret:
            #print(error)
            break
        counter += 1
        #print(counter)

        # This is to prevent high CPU usage, can be adjusted or removed
        cv2.waitKey(1)
        
    # Release the VideoCapture object and close all windows
    cap.release()
    cv2.destroyAllWindows()
