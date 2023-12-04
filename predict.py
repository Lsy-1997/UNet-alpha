import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet, UNet3, UNet_alpha
from utils.utils import plot_img_and_mask
from tqdm import tqdm

Cityscapes_color = [[  0,  0,  0],[  0,  0,  0],[  0,  0,  0],[  0,  0,  0],[  0,  0,  0],[111, 74,  0],[ 81,  0, 81],[128, 64,128],[244, 35,232],[250,170,160],[230,150,140],[ 70, 70, 70],[102,102,156],[190,153,153],[180,165,180],[150,100,100],[150,120, 90],[153,153,153],[153,153,153],[250,170, 30],[220,220,  0],[107,142, 35],[152,251,152],[ 70,130,180],[220, 20, 60],[255,  0,  0],[  0,  0,142],[  0,  0, 70],[  0, 60,100],[  0,  0, 90],[  0,  0,110],[  0, 80,100],[  0,  0,230],[119, 11, 32],[  0,  0,142]]
psv_dataset_color = [[0,0,0],[255,0,255],[0,0,255],[0,255,0],[255,0,0],[255,255,255],[0,255,255],[255,255,0],[255,128,128],[128,128,0]]

def plot_mask(img, masks, colors=None, alpha=0.5) -> np.ndarray:
    """Visualize segmentation mask.

    Parameters
    ----------
    img: numpy.ndarray
        Image with shape `(H, W, 3)`.
    masks: numpy.ndarray
        Binary images with shape `(N, H, W)`.
    colors: numpy.ndarray
        corlor for mask, shape `(N, 3)`.
        if None, generate random color for mask
    alpha: float, optional, default 0.5
        Transparency of plotted mask

    Returns
    -------
    numpy.ndarray
        The image plotted with segmentation masks, shape `(H, W, 3)`

    """
    masks = np.array(masks)
    img = np.array(img)
    colors = np.array(colors)
    if colors is None:
        colors = np.random.random((masks.shape[0], 3)) * 255
    else:
        if colors.shape[0] < masks.shape[0]:
            raise RuntimeError(
                f"colors count: {colors.shape[0]} is less than masks count: {masks.shape[0]}"
            )
    for mask, color in zip(masks, colors):
        mask = np.stack([mask, mask, mask], -1)
        img = np.where(mask, img * (1 - alpha) + color * alpha, img)

    return Image.fromarray(img.astype(np.uint8))

def draw_mask(mask, palette):
    result = Image.fromarray(mask.astype(np.uint8)).convert('P')
        
    result.putpalette(np.array(palette, dtype=np.uint8))
    
    return result

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='unet.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', type=str, metavar='INPUT', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=10, help='Number of classes')
    
    return parser.parse_args()


def get_output_filenames(out_dir, in_files):
    out_files = []
    for file in in_files:
        out_file = os.path.join(out_dir, file)
        out_files.append(out_file)
    return out_files


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((len(mask_values),mask.shape[-2], mask.shape[-1]), dtype=bool)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    # for i, v in enumerate(mask_values):
    #     if i ==0 :
    #         continue
    #     out[mask == i] = True 
    
    for i in mask_values:
        if i==0:
            continue
        out[i][mask==i] = True

    return out


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_dir = args.input
    if os.path.isdir(in_dir):
        in_files = os.listdir(in_dir)
        in_files_path = [os.path.join(in_dir, file) for file in in_files]
    else:
        in_files = in_dir
    
    n_classes = args.classes
        
    out_dir = args.output
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    out_files_path= get_output_filenames(out_dir, in_files)

    # net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    net = UNet_alpha(n_channels=3, n_classes=n_classes, bilinear=False, alpha=0.5)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)
    
    logging.info('Model loaded!')

    for i, filename in enumerate(tqdm(in_files_path)):
        # logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_filename = out_files_path[i]
            result = mask_to_image(mask, mask_values)
            # result_overlap = plot_mask(img=img, masks=result, colors=[[0,0,0],[0,255,0],[255,0,0],[0,0,255],[255,255,0],[0,255,255]])
            result_overlap = plot_mask(img=img, masks=result, colors=Cityscapes_color,alpha=1.0)
            
            drawed_mask = draw_mask(mask=mask, palette=Cityscapes_color)
            
            # result_overlap.save(out_filename)
            drawed_mask.save(out_filename)
            
            # logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)
