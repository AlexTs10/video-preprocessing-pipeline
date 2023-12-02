import os 
from lama_cleaner.model.lama import LaMa
from lama_cleaner.schema import Config
import torch 
import cv2
from schema import SOURCE_VIDEO_DIR, OUTOUT_FOLDER

    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# set to GPU for faster inference
model = LaMa(device=device)

for video_name in os.listdir(SOURCE_VIDEO_DIR):

    # loop thhrough all videos in raw_data folder 
    video = video_name.replace('.mp4', '')
    video_source = os.path.join(SOURCE_VIDEO_DIR, video)
    video_path = os.path.join(OUTOUT_FOLDER, video)
    boxes_path = os.path.join(video_path, 'boxes')
    original_frames_path = os.path.join(video_path, 'original_frames')
    ocr_gdino_frames_path = os.path.join(video_path, 'ocr_gdino_frames')
    sam_masks_path = os.path.join(video_path, 'sam_masks_frames')
    high_pass_filter_masks_path = os.path.join(video_path, 'high_pass_filter_masks_frames')        

    final_frames = os.path.join(video_path, 'final_frames')        
    os.makedirs(final_frames, exist_ok=True)


    l1 = len(os.listdir(sam_masks_path))
    l2 = len(os.listdir(high_pass_filter_masks_path))

    if l1 != l2:
        print('Error different nubmer of High pass and SAM masks. Check again!')
        break 

    for i in range(l1):

            
        # Read mask 1
        mask_sam = cv2.imread(os.path.join(sam_masks_path, f'{i}_.png'),
                              cv2.IMREAD_GRAYSCALE)

        # Read mask 2 
        mask_high_pass = cv2.imread(os.path.join(high_pass_filter_masks_path, f'{i}_.png'),
                                    cv2.IMREAD_GRAYSCALE)
        
        # Combine masks - OR
        # Perform bitwise OR operation
        final_mask = cv2.bitwise_or(mask_sam, mask_high_pass)
        
        # Read image 
        image = cv2.imread(os.path.join(original_frames_path, f'{i}_.png'),
                           cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Inpaint model Forward pass 
        result = model(rgb_image, final_mask, Config(hd_strategy="Original", 
                                   ldm_steps=25, 
                                   hd_strategy_crop_margin=196, 
                                   hd_strategy_crop_trigger_size=800, 
                                   hd_strategy_resize_limit=800))
    
        # Save image 
        cv2.imwrite(os.path.join(final_frames, f'{i}_.png'), result)

