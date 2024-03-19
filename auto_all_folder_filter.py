from IPython.display import display, HTML
display(HTML(
"""
<a target="_blank" href="https://colab.research.google.com/github/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
"""
))
from typing import Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import shutil
from pathlib import Path
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])  #调节透明度 越大越不透明
        img[m] = color_mask
    ax.imshow(img)

'''def save_mask(anns, image, path, basename):
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    
    for i, ann in enumerate(sorted_anns):
        mask = ann['segmentation']
        mask = np.stack([mask]*3, axis=-1)

        img = image.copy().astype(float)  # Change image data type to float
        cover = np.ones_like(img) * 255
        cover = cover * mask

        cover = cover.astype(np.uint8)
        img = img.astype(np.uint8)
        #result = cv2.addWeighted(img, 0.6, cover, 0.4, 0)
        result = cv2.addWeighted(img, 0.6, cover, 1, 0)
    

        # Add the mask to the image
       
        
        # Save the image\
        cv2.imwrite(f'{path}/mask_{i}.png', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))'''


'''def save_mask(anns, image, path, basename):
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    for i, ann in enumerate(sorted_anns):
        mask = ann['segmentation']
        mask = np.stack([mask]*3, axis=-1)

        img = image.copy().astype(float)  # Change image data type to float

        img_masked = img * mask # Part of the image where the mask is applied.
        img_unmasked = img * (1 - mask) # Part of the image where the mask is not applied.

        img_unmasked = img_unmasked * 0.5  # Reduce brightness by multiplying by a factor less than 1.

        # Combine the masked and unmasked parts.
        img_combined = img_masked + img_unmasked

        # Save the image
        cv2.imwrite(f'{path}/mask_{i}.png', cv2.cvtColor(img_combined.astype(np.uint8), cv2.COLOR_RGB2BGR))'''

def remove_small_regions(mask: np.ndarray, area_thresh: float, mode: str) -> Tuple[np.ndarray, bool]:
    """
    Removes small disconnected regions and holes in a mask. Returns the
    mask and an indicator of if the mask has been modified.
    """
    assert mode in ["holes", "islands"]
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]  # Row 0 is background label
    #print('sizes', sizes)
    #Dprint('enumerate(sizes)',enumerate(sizes))
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    if len(small_regions) == 0:
        return mask, False
    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        # If every region is below threshold, keep largest
        if len(fill_labels) == 0:
            fill_labels = [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask, True

#保存成黑白
def save_mask(anns, image, path, basename):

    #sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    for i, ann in enumerate(anns):
        #a = ann['original_index']
        mask = ann['segmentation']
        mask = np.stack([mask]*3, axis=-1)   #如果不进行remove处理，这句不用注释

        img = (mask*255).astype(np.uint8)  # Setting mask as white
        #processed_mask, modified = remove_small_regions(img, area_thresh=200, mode='holes')
        #processed_mask1, modified1 = remove_small_regions(img, area_thresh=200, mode='islands')
        #if modified==False and modified1==False:
            #cv2.imwrite(f'{path}/mask_{i}.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f'{path}/mask_{i}.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

'''def filter_small_masks(masks, min_area):
    """
    Filters out masks that are smaller than the given area in a list of mask dictionaries.

    Args:
    masks (list of dicts): The list of mask dictionaries to filter.
    min_area (int): The minimum area for a mask to keep.

    Returns:
    list of dicts: The list of filtered mask dictionaries.
    """
    # Filter out dictionaries whose area is less than min_area
    filtered_masks = [mask_dict for mask_dict in masks if mask_dict['area'] >= min_area]
    
    return filtered_masks'''
import datetime
now = datetime.datetime.now()
now_folder = now.strftime('%Y%m%d%H%M')
out_folder = 'F:\doctor\strawberry\segment-anything-main\segment-anything-main\output'
folder_path = os.path.join(out_folder, now_folder) 


sam_checkpoint = "F:\\doctor\\strawberry\\segment-anything-main\\segment-anything-main\\checkpoint\\sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

imgdir_path = 'F:\doctor\strawberry\segment-anything-main\segment-anything-main\img2'
#imgdir_path = 'F:\doctor\strawberry\segment-anything-main\segment-anything-main\img'   #原来的路径 大批量处理
files = os.listdir(imgdir_path)
for file in files:
    image_path = os.path.join(imgdir_path, file)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    basename = Path(image_path).stem  #提取无后缀的文件名
    os.makedirs(f'{folder_path}/{basename}', exist_ok=True)
    path_stem = f'{folder_path}/{basename}'
    # Create folder for masks

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator_2 = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=16,
    pred_iou_thresh=0.90,
    stability_score_thresh=0.95,
    stability_score_offset = 1.0,
    box_nms_thresh = 0.2,
    crop_n_layers=0,
    crop_nms_thresh = 0.7,
    crop_overlap_ratio = 500 / 1500,
    crop_n_points_downscale_factor=0,
    min_mask_region_area=200,  # Requires open-cv to run post-processing
)
    args = mask_generator_2.__dict__

    # Construct the content string by joining the arguments and their values
    content = 'mask_generator_2 = SamAutomaticMaskGenerator(\n'
    for key, value in args.items():
        content += f'    {key}={value},\n'
    content += ')'

    masks2 = mask_generator_2.generate(image)
    #masks2 = filter_small_masks(masks2_1, 30*30)

    save_mask(masks2, image, path_stem, basename)
    #保存

    # Plot all masks on the image
    plt.figure(figsize=(20,20))
    plt.imshow(image)
    show_anns(masks2)
    plt.axis('off')
    plt.savefig(os.path.join(folder_path, basename,f"{basename}_all_masks.png"))
    file_name = os.path.join(folder_path, f"parameter.txt")
    with open(file_name, 'w') as file:
        file.write(content)
    print('Done!')

    #plt.show() 
    #python auto_all_folder_filter.py