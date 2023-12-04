import torch

def calculate_iou(pred_mask, true_mask):
    intersection = torch.logical_and(pred_mask, true_mask).sum()
    union = torch.logical_or(pred_mask, true_mask).sum()
    epsilon = 1e-6
    iou = (intersection.float()+epsilon) / (union.float()+epsilon)
    return iou

def calculate_miou(pred_masks, true_masks):
    num_classes = pred_masks.shape[1]
    ious = []
    for class_idx in range(num_classes):
        pred_mask = pred_masks[:, class_idx, :, :]
        true_mask = true_masks[:, class_idx, :, :]
        iou = calculate_iou(pred_mask, true_mask)
        ious.append(iou)
    miou = torch.mean(torch.tensor(ious))
    
    return miou, torch.tensor(ious)
