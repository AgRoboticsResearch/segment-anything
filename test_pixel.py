import cv2
import numpy as np
from typing import Tuple

# 加载图片
image = cv2.imread('F:\doctor\strawberry\segment-anything-main\segment-anything-main\output/202403182201/1\mask_22.png', cv2.IMREAD_GRAYSCALE)

# 计算白色像素的数量
# cv2.countNonZero函数会统计参数中非零（因此，黑色像素不会被统计）的像素总数。由于我们的图像是二值图像，白色像素值为255，因此这个函数将计算出所有的白色像素。
white_pixels = cv2.countNonZero(image)

print(f'Number of white pixels: {white_pixels}')

import cv2
import numpy as np

#def remove_small_regions(mask: np.ndarray, area_thresh: float, mode: str) -> np.ndarray, bool:
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
    print('sizes', sizes)
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

# Load the image as grayscale
mask_3 = cv2.imread('F:\doctor\strawberry\segment-anything-main\segment-anything-main\output/202403181634/1\mask_18.png', cv2.IMREAD_GRAYSCALE)

# Call the remove_small_regions function
processed_mask, modified = remove_small_regions(mask_3, area_thresh=200, mode='holes')
processed_mask1, modified1 = remove_small_regions(mask_3, area_thresh=200, mode='islands')

# If the mask was modified, save the new mask
if modified:
    print('modified',modified)
    cv2.imwrite("F:\doctor\processed_mask_3.png", processed_mask*255)

if modified1:
    print('modified1',modified1)
    cv2.imwrite("F:\doctor\processed_mask_3.png", processed_mask*255)


    '''@staticmethod
    def postprocess_small_regions(
        mask_data: MaskData, min_area: int, nms_thresh: float,
    ) -> MaskData:
        """
        Removes small disconnected regions and holes in masks, then reruns
        box NMS to remove any new duplicates.

        Edits mask_data in place.

        Requires open-cv as a dependency.
        """
        #print('line 342 运行了postprocess_small_regions函数')
        if len(mask_data["rles"]) == 0:
            print('len(mask_data["rles"]) == 0')
            return mask_data

        # Filter small disconnected regions and holes
        mask_data["original_indices"] = list(range(len(mask_data["rles"])))

        new_masks = []
        scores = []

        for i, rle in enumerate(mask_data["rles"]):
            mask = rle_to_mask(rle)
            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append((torch.as_tensor(mask).unsqueeze(0), i))  # Append as a tuple
            scores.append((float(unchanged), i))  # Append score with index as a tuple

        # Extract relevant tensors and data for nms
        nms_boxes = torch.cat([mask for mask, i in new_masks], dim=0)
        nms_scores = torch.tensor([score for score, i in scores])
        nms_categories = torch.zeros_like(nms_boxes[:, 0])

        keep_by_nms = batched_nms(
            nms_boxes.float(),
            nms_scores,
            nms_categories,
            iou_threshold=nms_thresh,
        )

        # Only recalculate RLEs for masks that have changed
        for idx in keep_by_nms:
            if scores[idx][0] == 0.0:  # Use the score from the tuple
                mask_torch, mask_original_index = new_masks[idx]  # Extract the mask and original index
                mask_data["rles"][mask_original_index] = mask_to_rle_pytorch(mask_torch)[0]
                mask_data["boxes"][mask_original_index] = nms_boxes[idx]  # Update res directly

        indices_to_keep = [scores[idx][1] for idx in keep_by_nms.tolist()]
        mask_data['original_indices'] = indices_to_keep
        mask_data.filter(indices_to_keep)

        return mask_data'''