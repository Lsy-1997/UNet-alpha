import torch
import torch.nn.functional as F
from torchsummary import summary

import numpy as np
from tqdm import tqdm
import os
from torch.utils.data import DataLoader, random_split
from utils.data_loading import BasicDataset, CarvanaDataset
import random
import time
from thop import profile
from unet import UNet, UNet3, UNet_alpha

from utils.dice_score import multiclass_dice_coeff, dice_coeff
from utils.mIOU import calculate_miou

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    miou = 0
    count = 0
    cls_iou = [0.0 for i in range(net.n_classes)]
    cls_iou = torch.tensor(cls_iou)
    
    # iterate over the validation set
    # with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
    with torch.autocast(device.type, enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            count += 1
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                miou_tmp, cls_iou_tmp = calculate_miou(mask_pred, mask_true)
                miou += miou_tmp
                cls_iou += cls_iou_tmp
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
                miou_tmp, cls_iou_tmp = calculate_miou(mask_pred, mask_true)
                miou += miou_tmp
                cls_iou += cls_iou_tmp

    net.train()
    return dice_score / max(num_val_batches, 1), miou / count, cls_iou / count



if __name__ == '__main__':
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    data_path = './data/woodscape'
    dir_img = os.path.join(data_path,"images")
    dir_mask = os.path.join(data_path,"labels")
    img_scale = 0.5
    val_percent = 0.1
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    # try:
    #     dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    # except (AssertionError, RuntimeError, IndexError):
    dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    # loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    loader_args = dict(batch_size=1, num_workers=12, pin_memory=True)
    
    # train_loader = DataLoader(train_set, shuffle=True, worker_init_fn=np.random.seed(seed) ,**loader_args)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    
    model = UNet_alpha(n_channels=3, n_classes=10, bilinear=False, alpha=2)
    model = model.to(memory_format=torch.channels_last)
    
    state_dict_path = './checkpoints/UNet_alpha2.0_woodscape/checkpoint_epoch100.pth'
    state_dict = torch.load(state_dict_path, map_location=device)
    # del state_dict['mask_values']
    model.load_state_dict(state_dict, strict=False)
    
    model.to(device=device)
    
    val_score, val_miou, cls_ious = evaluate(model, val_loader, device, amp=False)
    
    print(cls_ious)