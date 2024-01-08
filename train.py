import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.models as models

from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
from datetime import datetime

import wandb
import json
from PIL import Image

from evaluate import evaluate
from unet import UNet, UNet3, UNet_alpha
from utils.data_loading import BasicDataset, CarvanaDataset, TongjiParkingDataset
from utils.dice_score import dice_loss

from swin_transformer import SwinTransformer, swin_t_upernet, swin_t_upernet_pretrained, swin_t_upernet_pretrained_rgbi

from fcn import FCN_resnet50
# from models.ffnet_S_mobile import segmentation_ffnet86S_dBBB_mobile_lsy

# dir_img = Path('./data/imgs/')
# dir_mask = Path('./data/masks/')


#设置全局随机种子，使结果更具重复性
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def train_model(
        model,
        device,
        data_path,
        dir_checkpoint,
        alpha,
        use_wandb: bool = True ,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        use_intensity: int = 0,
        filter_size: int = 1,
        use_diceloss: int = 1
):
    # 1. Create dataset
    dir_img = os.path.join(data_path,"images_rgbi")
    # else:
    #     dir_img = os.path.join(data_path,"images")
    dir_mask = os.path.join(data_path,"labels")
    train_image_dir = os.path.join(dir_img, "train")
    train_mask_dir = os.path.join(dir_mask, "train")
    val_image_dir = os.path.join(dir_img, "val")
    val_mask_dir = os.path.join(dir_mask, "val")

    # try:
    #     dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    # except (AssertionError, RuntimeError, IndexError):
    # dataset = BasicDataset(dir_img, dir_mask, img_scale)
    train_set = TongjiParkingDataset(train_image_dir, train_mask_dir, img_scale, filter_size, use_intensity)
    val_set = TongjiParkingDataset(val_image_dir, val_mask_dir, img_scale, filter_size, use_intensity)
    # 2. Split into train / validation partitions
    # n_val = int(len(dataset) * val_percent)
    # n_train = len(dataset) - n_val
    # train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    n_train = len(train_set)
    n_val = len(val_set)

    # 3. Create data loaders
    # loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    loader_args = dict(batch_size=batch_size, num_workers=12, pin_memory=True)
    
    # train_loader = DataLoader(train_set, shuffle=True, worker_init_fn=np.random.seed(seed) ,**loader_args)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    if use_wandb:
        experiment = wandb.init(project='RGBI-semantic-segmentation', resume='allow', anonymous='must', entity='tj_cvrsg')
        experiment.config.update(
            dict(epochs=epochs, alpha = alpha, batch_size=batch_size, learning_rate=learning_rate,
                val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
        )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Alpha:           {alpha}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
        Use Intensity: {use_intensity}
        Use Diceloss: {use_diceloss}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    # optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, capturable=True)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=20)  # goal: maximize Dice score
    scheduler = optim.lr_scheduler.LinearLR(optimizer, learning_rate)  # goal: maximize Dice score
    # scheduler = optim.lr_scheduler.PolynomialLR(optimizer, learning_rate)  # goal: maximize Dice score
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        if use_diceloss==1:
                            loss += dice_loss(
                                F.softmax(masks_pred, dim=1).float(),
                                F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                                multiclass=True
                            )

                optimizer.zero_grad(set_to_none=True)
                # grad_scaler.scale(loss).backward()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                # grad_scaler.step(optimizer)
                optimizer.step()
                # grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                if use_wandb:
                    experiment.log({
                        'train loss': loss.item(),
                        'step': global_step,
                        'epoch': epoch
                    })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (1 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not torch.isinf(value).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if value.grad != None:
                                if not torch.isinf(value.grad).any():
                                    histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        # train_score, train_miou = evaluate(model, train_loader, device, amp)
                        val_score, val_miou, cls_ious = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        if use_wandb:
                            true_mask_rgb = mask_coloring(true_masks[0].float().cpu())
                            pred_mask_rgb = mask_coloring(masks_pred.argmax(dim=1)[0].float().cpu())
                            if use_intensity == 1:
                                # wandb_images = images[0].cpu()[0:3,:,:]
                                wandb_images = images[0].cpu()
                                # wandb_intensity = images[0].cpu()[3,:,:]
                                # wandb_intensity[wandb_intensity==0] = 1
                                wandb_images[3,:,:][wandb_images[3,:,:]==0] = 1
                            elif use_intensity == 0:
                                wandb_images = images[0].cpu()
                            elif use_intensity == 2:
                                wandb_images = images[0].cpu()
                            try:
                                experiment.log({
                                    'learning rate': optimizer.param_groups[0]['lr'],
                                    'validation Dice': val_score,
                                    'val miou': val_miou,
                                    # 'train Dice': train_score,
                                    'images': wandb.Image(wandb_images),
                                    'masks': {
                                        # 'true': wandb.Image(true_masks[0].float().cpu()),
                                        # 'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                        'true': wandb.Image(true_mask_rgb),
                                        'pred': wandb.Image(pred_mask_rgb),
                                    },
                                    'step': global_step,
                                    'epoch': epoch,
                                    **histograms
                                })
                            except:
                                pass

        if save_checkpoint and epoch%10==0:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            # state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, os.path.join(dir_checkpoint, 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')
        if epoch==epochs:
            print('classes iou:\n')
            print(cls_ious)
            
def mask_coloring(mask):
    label_color_file = 'data/tongji_parking_rgbi/label_color.json'
    mask_size = mask.shape
    with open(label_color_file, 'r') as file:
        category_colors = json.load(file)
        category_colors = {int(k): tuple(v) for k, v in category_colors.items()}
    color_mask = np.zeros((mask_size[0], mask_size[1], 3), dtype=np.uint8)
    # color_mask = torch.zeros((mask_size[0], mask_size[1], 3), dtype=torch.uint8)
    for value, color in category_colors.items():
        color_mask[mask == value] = color
    color_mask_image = Image.fromarray(color_mask)
    
    return color_mask_image

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--data-path', '-d', dest='data_path', metavar='D', default='data/PSV_dataset', type=str, help='Path of dataset')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=20.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=6, help='Number of classes')
    
    parser.add_argument('--model', type=str ,default='UNet_alpha')
    parser.add_argument('--alpha', type=float ,default=1)
    parser.add_argument('--use-wandb', type=int, default=1)
    parser.add_argument('--use-intensity', type=int, default=1)
    parser.add_argument('--intensity-upsample-filter-size', type=int, default=1)
    parser.add_argument('--use-diceloss', type=int, default=1)
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    dataset = args.data_path.split('/')[-1]
    current_date = datetime.now().date()
    
    if args.model == 'UNet_alpha':
        dir_checkpoint = os.path.join('./checkpoints', args.model + str(args.alpha) + '_' + dataset + f'_useintensity{args.use_intensity}_' + f'fs{args.intensity_upsample_filter_size}_' + str(current_date))
    else:    
        dir_checkpoint = os.path.join('./checkpoints', args.model + '_' + dataset + str(current_date))
    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    if args.use_intensity == 0:
        channels = 3
    elif args.use_intensity == 1:
        channels = 4
    elif args.use_intensity == 2:
        channels = 1
    
    # model = models.swin_v2_t(weights=models.Swin_V2_T_Weights)
        
    if args.model == 'UNet':
        model = UNet(n_channels=channels, n_classes=args.classes, bilinear=args.bilinear)
    elif args.model == 'UNet3':
        model = UNet3(n_channels=channels, n_classes=args.classes, bilinear=args.bilinear)
    elif args.model == 'UNet_alpha':
        if args.use_intensity == 1:
            model = UNet_alpha(n_channels=channels, n_classes=args.classes, bilinear=args.bilinear, alpha=args.alpha)
    elif args.model == 'fcn':    
        model = FCN_resnet50(n_channels=channels, n_classes=args.classes)
    elif args.model == 'SwinTransformer':
        # 预训练
        # swin_v2_t = models.swin_v2_t(weights=models.Swin_V2_T_Weights)
        model = swin_t_upernet(
                                    hidden_dim=96,
                                    layers=(2, 2, 6, 2),
                                    heads=(3, 6, 12, 24),
                                    channels=channels,
                                    num_classes=args.classes,
                                    head_dim=32,
                                    window_size=8,
                                    downscaling_factors=(4, 2, 2, 2),
                                    relative_pos_embedding=True,
                                    size=[512, 512]
                                )
    elif args.model == 'SwinTransformer_pretrained':
        model = swin_t_upernet_pretrained(num_classes=args.classes)
    elif args.model == 'swin_t_pretrain_rgbi':
        model = swin_t_upernet_pretrained_rgbi(num_classes=args.classes)
    elif args.model == 'resnet':
        model = models.resnet50(models.ResNet50_Weights)
        
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{channels} input channels\n'
                 f'\t{args.classes} output channels (classes)\n')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        if 'mask_values' in state_dict:
            del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            dir_checkpoint=dir_checkpoint,
            alpha=args.alpha,
            use_wandb=args.use_wandb,
            data_path=args.data_path,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            use_intensity = args.use_intensity,
            filter_size = args.intensity_upsample_filter_size,
            use_diceloss=args.use_diceloss
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            dir_checkpoint=dir_checkpoint,
            alpha=args.alpha,
            use_wandb=args.use_wandb,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
