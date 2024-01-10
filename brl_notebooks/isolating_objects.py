import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import shutil
from PIL import Image
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

class Maskdata:
    def __init__(self, mask, score, idx):
        self.mask = mask
        self.score = score
        self.idx = idx

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.random.uniform(low=0.3, high=1.0, size=3)
        color= color / np.max(color)
        color = np.concatenate([color, np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def nms(masks_list, conf_threshold = 0.9, lap_threshold=0.7):
    # filter masks based on confidence threshold and overlap
    mask_list_thresholded= []
    mask_list_new = []
    masks_sorted = sorted(masks_list, key=lambda x: x.score, reverse=True)

    for mask_data in masks_sorted:
        if mask_data.score > conf_threshold:
            mask_list_thresholded.append(mask_data)
        else:
            pass
    
    while len(mask_list_thresholded) > 0:
        current_mask = mask_list_thresholded[0]
        mask_list_thresholded = mask_list_thresholded[1:]
        mask_list_new.append(current_mask)

        mask_list_thresholded = [
        mask_data for mask_data in mask_list_thresholded
        if calculate_overlap(current_mask.mask, mask_data.mask) <= lap_threshold
        ]
        
    print("mask_list_new:", len(mask_list_new))
    return mask_list_new

def calculate_overlap(mask, masks):
    # Calculate overlap between two masks
    overlap_pixels = (mask & masks).sum()
    total_pixels = min(mask.sum(), masks.sum())
    overlap = overlap_pixels / total_pixels
    return overlap 


if __name__ == "__main__":
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    device = "cuda"
    model_type = "vit_h"
    input_folder = "/home/rcli/sam-ws/segment-anything/brl_notebooks/images"
    save_folder = "/home/rcli/sam-ws/segment-anything/brl_notebooks/isolating_masks/Attempt_2"

    files = os.listdir(input_folder)
    for file in files:
        image_path = os.path.join(input_folder, file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        predictor.set_image(image)

        image_height = image.shape[0]
        image_width = image.shape[1]
        image_size = image_height * image_width
        grid_size = 21 # prompt点间距
        x_points = np.arange(21,image_width,grid_size)
        y_points = np.arange(21,image_height,grid_size)
        grid_points = [(x, y) for x in x_points for y in y_points]

        masks_list = []
        index = 1
        for i in range(len(grid_points)):
            input_point = np.array([grid_points[i]])
            input_label = np.array([1])
            masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True, 
            )

            mask = masks[1]
            score = scores[1]
            mask_data = Maskdata(mask, score, index)

            masks_list.append(mask_data)
            index += 1

        sam_mask_list = nms(masks_list, conf_threshold=0.88, lap_threshold=0.7)

        output_subfolder = os.path.join(save_folder, os.path.splitext(file)[0])
        os.makedirs(output_subfolder, exist_ok=True)

        fig_idx = 0
        for mask_data in sam_mask_list:
            # if 1000 < mask_data.mask.sum() <= 100000:
            print("mask_size:", mask_data.mask.sum())

            mask_image = (mask_data.mask * 255).astype(np.uint8)
            mask_image[mask_data.mask == 0] = 64
            isolated = np.dstack([image, mask_image])

            fig_idx += 1
            plt.figure(figsize=(10,10))
            plt.imshow(isolated)
            # plt.title(f"isolated object {fig_idx}", fontsize=18)
            plt.axis('off')
            output_file = os.path.join(output_subfolder, f"{fig_idx}.png")
            plt.savefig(output_file, bbox_inches="tight", dpi=300, pad_inches=0.0)
            plt.close()
    

