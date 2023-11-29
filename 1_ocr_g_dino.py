import os
import numpy as np
import torch
import cv2

from utils import load_model_hf, grounding_dino, ocr

# Use this command for evaluate the Grounding DINO model
# Or you can download the model by yourself
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)

## PART 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

source_video_dir = 'sun_videos_cut'
output_folder = 'processed'
os.makedirs(output_folder, exist_ok=True)


for video_name in os.listdir(source_video_dir):

        
    # loop through all videos in raw_data folder 
    video = video_name.replace('.mp4', '')

    video_source = os.path.join(source_video_dir, video)

    video_path = os.path.join(output_folder, video)
    os.makedirs(video_path, exist_ok=True)

    boxes_path = os.path.join(video_path, 'boxes')
    os.makedirs(boxes_path, exist_ok=True)

    original_frames_path = os.path.join(video_path, 'original_frames')
    os.makedirs(original_frames_path, exist_ok=True)

    ocr_gdino_frames_path = os.path.join(video_path, 'ocr_gdino_frames')
    os.makedirs(ocr_gdino_frames_path, exist_ok=True)

    sam_masks_path = os.path.join(video_path, 'sam_masks_frames')
    os.makedirs(sam_masks_path, exist_ok=True)

    high_pass_filter_masks_path = os.path.join(video_path, 'high_pass_filter_masks_frames')
    os.makedirs(high_pass_filter_masks_path, exist_ok=True)
        

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
            print(error)
            break
        cv2.imwrite(os.path.join(original_frames_path, f'{counter}_.png'), frame)

        frame = ocr(frame)
        # Save the image as a png file
        temp_path = 'temp_frame.png'
        cv2.imwrite(temp_path, frame)

        boxes, logits, phrases, image_source  = grounding_dino(image_path=temp_path, groundingdino_model=groundingdino_model , device=device)
        cv2.imwrite(os.path.join(ocr_gdino_frames_path, f'{counter}_.png'), image_source)

        # Save the tensor to a file
        torch.save(boxes, os.path.join(boxes_path, f'{counter}_.pt'))
        counter += 1

        # This is to prevent high CPU usage, can be adjusted or removed
        cv2.waitKey(1)
        
    # Release the VideoCapture object and close all windows
    cap.release()
    cv2.destroyAllWindows()
