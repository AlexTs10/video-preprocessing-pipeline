import os, sys

sys.path.append(os.path.join(os.getcwd(),
                              "GroundingDINO"))

import argparse
import os
import copy
from matplotlib import pyplot as plt

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict

import supervision as sv
from huggingface_hub import hf_hub_download

import cv2
import pytesseract
from pytesseract import Output
from segment_anything import build_sam, SamPredictor 

def load_model_hf(repo_id, filename, ckpt_config_filename, device):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model   

def grounding_dino(image_path, groundingdino_model, device):

#    TEXT_PROMPT = "redaction bars"
    TEXT_PROMPT = "digital annotations . text . logos"
    BOX_TRESHOLD = 0.3
    TEXT_TRESHOLD = 0.25
    
    #image_source, image = load_image(local_image_path)
    image_source, image = load_image(image_path)
    
    boxes, logits, phrases = predict(
        model=groundingdino_model, 
        image=image, 
        caption=TEXT_PROMPT, 
        box_threshold=BOX_TRESHOLD, 
        text_threshold=TEXT_TRESHOLD,
        device=device
    )

    #print(boxes, logits, phrases )

    return boxes, logits, phrases, image_source 


def ocr(image):
        
    # Convert the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use thresholding to get a binary image
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Use pytesseract to get the bounding boxes of text regions
    custom_config = r'--oem 3 --psm 6'
    details = pytesseract.image_to_data(binary, output_type=Output.DICT, config=custom_config, lang='eng')
    
    # Loop over each text region and mask it
    masked_image = image.copy()
    for i in range(len(details['text'])):
        if int(details['conf'][i]) > 60:  # Confidence threshold.
            (x, y, w, h) = (details['left'][i], details['top'][i], details['width'][i], details['height'][i])
            
            # Draw a rectangle with the background color
            cv2.rectangle(masked_image, (x, y), (x + w, y + h), (255, 255, 255), -1)  # White color mask

            # Print the text
            #print(details['text'][i])
    return masked_image



def load_sam(device):
    sam_checkpoint = 'sam_vit_h_4b8939.pth'
    #sam_checkpoint = 'sam_vit_b_01ec64.pth'
    sam = build_sam(checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)
    
    return sam_predictor

def sam_inference(image_source, boxes, sam_predictor):
    # set image
    sam_predictor.set_image(image_source)
    
    # box: normalized box xywh -> unnormalized xyxy
    H, W, _ = image_source.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(device)

    masks, _, _ = sam_predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes,
            multimask_output = False,
        )
    return masks

def show_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

def padding_mask(image_mask,  kernel_size=7):
    
    # Define the padding size (dilation size)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Dilate the entire mask
    dilated_mask = cv2.dilate(image_mask.astype(np.uint8), kernel, iterations=1)
    
    # Find contours in the dilated mask
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create an empty mask to draw the padded contours
    padded_mask = np.zeros_like(image_mask)
    
    # Draw the contours on the new mask
    for contour in contours:
        cv2.drawContours(padded_mask, [contour], -1, 1, thickness=cv2.FILLED)

    return padded_mask


def dilate_mask(mask, dilate_factor=10):
    mask = mask.astype(np.uint8)
    mask = cv2.dilate(
        mask,
        np.ones((dilate_factor, dilate_factor), np.uint8),
        iterations=1
    )
    return mask


def high_pass_filter_mask(image_path):

    image = cv2.imread(image_path)

    # High-Pass Filter: 
    low_pass = cv2.GaussianBlur(image, (9, 9), 4)
    
    # Then, we subtract the blurred image from the original grayscale image to get the high pass filtered image.
    high_pass = cv2.subtract(image, low_pass)
    
    # Create a binary mask from the high-frequency image
    _, mask = cv2.threshold(high_pass, 15, 255, cv2.THRESH_BINARY)
    
    # Convert the mask to grayscale (single channel)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    return mask

