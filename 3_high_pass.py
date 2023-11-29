import os
import numpy as np
import cv2
from utils import high_pass_filter_mask


## PART 2
source_video_dir = 'sun_videos_cut'
output_folder = 'processed'
    
for video_name in os.listdir(source_video_dir):

    # loop thhrough all videos in raw_data folder 
    video = video_name.replace('.mp4', '')
    video_source = os.path.join(source_video_dir, video)
    video_path = os.path.join(output_folder, video)
    boxes_path = os.path.join(video_path, 'boxes')
    original_frames_path = os.path.join(video_path, 'original_frames')
    ocr_gdino_frames_path = os.path.join(video_path, 'ocr_gdino_frames')
    sam_masks_path = os.path.join(video_path, 'sam_masks_frames')
    high_pass_filter_masks_path = os.path.join(video_path, 'high_pass_filter_masks_frames')        


counter = 0 
for filename in sorted(os.listdir(original_frames_path)):
    frame_path = os.path.join(original_frames_path, filename)

    mask = high_pass_filter_mask(image_path=frame_path) # 0/255 mask - ready

    # Save the mask as a PNG file
    cv2.imwrite(os.path.join(high_pass_filter_mask, f'{counter}_.png'), mask)
    