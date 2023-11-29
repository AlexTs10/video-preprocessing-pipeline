import os
import numpy as np
import torch
import cv2
from utils import load_sam, sam_inference, dilate_mask

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


    box_dir = boxes_path
    frame_dir = os.path.join(output_folder, video, ocr_gdino_frames_path)

    sam_predictor = load_sam(device=device)
    counter=0
    # Loop through all files in the directory
    length = len(os.listdir(frame_dir))
    for i in range(length):
        
        # Construct the full file path
        frame_path = os.path.join(frame_dir, f'{i}_.png')
        box_path = os.path.join(box_dir, f'{i}_.pt')
        
        # Load the tensor
        boxes = torch.load(box_path)
        image = cv2.imread(frame_path)

        if boxes.numel() == 0: # empty
            # empty mask no need for sam
            mask = np.zeros((image.shape[0], image.shape[1])).astype('uint8')
            cv2.imwrite(os.path.join(sam_masks_path, f'{counter}_.png'), mask)

        else:
                

            # (#_masks, 1, W, H)
            masks = sam_inference(image_source=image, boxes=boxes, sam_predictor=sam_predictor) # True False Masks 

            # convert to 0/1 mask and perform OR on all masks to combine into 1
            int_mask = torch.any(masks, dim=0)[0].int().numpy() 

            # padd the mask
            mask = dilate_mask(int_mask) # still 0/1 mask
                
            # Convert the mask to an 8-bit image (0-255)
            mask_8bit = (mask * 255).astype('uint8')
            
            # Save the mask as a PNG file
            cv2.imwrite(os.path.join(sam_masks_path, f'{counter}_.png'), mask_8bit)
