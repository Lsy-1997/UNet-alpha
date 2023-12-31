import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet, UNet3, UNet_alpha
from swin_transformer import SwinTransformer, swin_t_upernet, swin_t_upernet_pretrained
from utils.utils import plot_img_and_mask
from tqdm import tqdm

import cv2
from scipy.ndimage import maximum_filter

tongji_parking_rgbi = [[0, 0, 0],[46,120,193],[100,238,87],[200,213,23],[11,116,231],[42,7,209]]
def plot_mask(img, masks, colors=None, alpha=0.5) -> np.ndarray:

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

def preprocess(mask_values, pil_img, scale, is_mask, square_pad=False, use_intensity = 0, filter_size = 1):
        img = np.asarray(pil_img)
        img = cv2.resize(img, (1024, 1024))
        
        w, h = img.shape[0:2]
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        
        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i
            
            return mask

        else:
            if use_intensity==0:
                img = img[:,:,0:3]
            elif use_intensity==1:
                rgb = img[:,:,0:3]
                intensity = img[:,:,3]
                filtered_intensity = maximum_filter(intensity, size=filter_size)
                filtered_intensity = filtered_intensity[...,None]
                img = np.concatenate((rgb, filtered_intensity),axis=2)
            elif use_intensity==2:
                intensity = img[:,:,3]
                filtered_intensity = maximum_filter(intensity, size=filter_size)
                filtered_intensity = filtered_intensity[...,None]
                img = filtered_intensity
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    if(net.n_channels==3):
        img = torch.from_numpy(preprocess(None, full_img, scale_factor, is_mask=False, use_intensity=0))
    elif (net.n_channels==4):
        img = torch.from_numpy(preprocess(None, full_img, scale_factor, is_mask=False, use_intensity=1))
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
    parser.add_argument('--classes', '-c', type=int, default=6, help='Number of classes')
    
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

def brightness_adjust(img, factor):
    
    brightened_image = cv2.add(img, np.array([factor]))
    
    return brightened_image

def add_serial_num(original_image, number):

    original_width, original_height = original_image.size

    padding_height = 80  # 设定 padding 的高度

    # 创建一个新的图像，高度等于原始图像高度加上 padding 高度
    new_image = Image.new("RGB", (original_width, original_height + padding_height), color="white")

    # 将原始图像复制到新图像上，位于 padding 下方
    new_image.paste(original_image, (0, padding_height))

    # 添加文本
    draw = ImageDraw.Draw(new_image)
    font_size = 80  # 字体大小
    font = ImageFont.truetype(font='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',size=font_size)  # 选择合适的字体和大小

    # 计算文本居中的位置
    text_width, text_height = font_size, font_size
    text_x = (original_width - text_width) // 2
    text_y = (padding_height - text_height) // 2

    # 在 padding 区域绘制编号
    
    draw.text((text_x, text_y), str(number), fill="black", font=font)

    return new_image


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # in_dir = args.input
    in_dir = "/home/cvrsg/lsy/Parking_line_detection/UNet-alpha/data/tongji_parking_rgbi_slice_splitted_truergbi/images_rgbi/val"
    if os.path.isdir(in_dir):
        in_files = os.listdir(in_dir)
        in_files_path = [os.path.join(in_dir, file) for file in in_files]
    else:
        in_files = in_dir
    
    n_classes = args.classes
        
    # out_dir = args.output
    factor = 100  # 增加或减少的亮度值
    out_dir = f"/home/cvrsg/lsy/Parking_line_detection/UNet-alpha/test/UNet_a0.5_RGBI_swin_t_compare_brightness_adjust/{factor}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    out_files_path= get_output_filenames(out_dir, in_files)
    
    labels_dir = '/home/cvrsg/lsy/Parking_line_detection/UNet-alpha/data/tongji_parking_rgbi_slice_splitted_truergbi/labels/val'
    labels_path = get_output_filenames(labels_dir, in_files)
    
    models_list = [
                    'checkpoints/UNet_alpha0.5_tongji_parking_rgbi_slice_splitted_truergbi_useintensity0_fs1_2024-01-04/checkpoint_epoch200.pth',
                    'checkpoints/UNet_alpha0.5_tongji_parking_rgbi_slice_splitted_truergbi_useintensity1_fs1_2024-01-04/checkpoint_epoch200.pth',
                    'checkpoints/UNet_alpha0.5_tongji_parking_rgbi_slice_splitted_truergbi_useintensity1_fs5_2024-01-04/checkpoint_epoch200.pth',
                    'checkpoints/UNet_alpha0.5_tongji_parking_rgbi_slice_splitted_truergbi_useintensity1_fs15_2024-01-04/checkpoint_epoch200.pth',
                    'checkpoints/SwinTransformer_pretrained_tongji_parking_rgbi_slice_splitted_truergbi2024-01-07/checkpoint_epoch200.pth'
    ]

    nets = []
    nets.append(UNet_alpha(n_channels=3, n_classes=n_classes, bilinear=False, alpha=0.5))
    nets.append(UNet_alpha(n_channels=4, n_classes=n_classes, bilinear=False, alpha=0.5))
    nets.append(UNet_alpha(n_channels=4, n_classes=n_classes, bilinear=False, alpha=0.5))
    nets.append(UNet_alpha(n_channels=4, n_classes=n_classes, bilinear=False, alpha=0.5))
    nets.append(swin_t_upernet_pretrained(num_classes=args.classes))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    for net in nets:
        net.to(device=device)

    state_dicts = []
    for model_path in models_list:
        state_dicts.append(torch.load(model_path, map_location=device))
        
    mask_values = [0, 1, 2, 3, 4, 5]
    
    for i in range(len(nets)):
        nets[i].load_state_dict(state_dicts[i])
    
    logging.info('Model loaded!')

    for i, filename in enumerate(tqdm(in_files_path)):
        img = Image.open(filename)
        label = Image.open(labels_path[i])
        
        img_np = np.asarray(img)
        
        label_np = np.asarray(label)
        img_np = cv2.resize(img_np, (1024, 1024))
        label_np = cv2.resize(label_np, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        
        img_rgb = img_np[:,:,0:3]
        img_rgb_ori = img_rgb
        img_intensity = img_np[:,:,3]
        img_intensity = img_intensity[..., None]
        
        # factor_array = np.full_like(img_rgb, factor, dtype=np.float32)
        factor_array = np.full_like(img_rgb, factor, dtype=np.uint8)
        img_rgb = np.array(img_rgb, dtype=np.float32)
        # img_rgb = cv2.add(img_rgb, factor_array)
        # img_rgb = cv2.multiply(img_rgb, factor_array)
        img_rgb = np.array(img_rgb, dtype=np.uint8)
        img_rgb = cv2.subtract(img_rgb, factor_array)
        
        img_rgbi = np.concatenate((img_rgb, img_intensity),axis=2)
        
        img_rgb = Image.fromarray(img_rgb)
        img_rgb_ori = Image.fromarray(img_rgb_ori)
        img_rgbi = Image.fromarray(img_rgbi)

        masks = []
        for net in nets:
            masks.append(predict_img(net=net, full_img=img_rgbi, scale_factor=args.scale, out_threshold=args.mask_threshold, device=device))


        out_filename = out_files_path[i]
        results = []
        for mask in masks:
            results.append(mask_to_image(mask, mask_values))
        
        label = mask_to_image(label_np, mask_values)
        
        results_overlap = []
        for result in results:
            results_overlap.append(plot_mask(img=img_rgb, masks=result, colors=tongji_parking_rgbi,alpha=1.0))
            
        groundtruth_overlap = plot_mask(img=img_rgb, masks=label, colors=tongji_parking_rgbi,alpha=1.0)
        
        # drawed_mask1 = draw_mask(mask=mask1, palette=tongji_parking_rgbi)
        # drawed_mask2 = draw_mask(mask=mask2, palette=tongji_parking_rgbi)
        # drawed_mask3 = draw_mask(mask=mask3, palette=tongji_parking_rgbi)
        # drawed_mask4 = draw_mask(mask=mask4, palette=tongji_parking_rgbi)
        
        total_width = img_rgb.width + groundtruth_overlap.width
        
        max_height = max(img_rgb.height, groundtruth_overlap.height)
        for result_overlap in results_overlap:
            total_width = total_width + result_overlap.width
            max_height = max(max_height, result_overlap.height)
            
        # 创建一个新图像
        new_im = Image.new('RGB', (total_width, max_height + 20)) # padding 20

        # 水平拼接图像
        x_offset = 0
        # white_space = np.array(255, (10,max_height))
        # total_width = total_width + white_space
        final_imgs_list = [img_rgb_ori, *results_overlap, groundtruth_overlap]
        for i in range(len(final_imgs_list)):
            img = final_imgs_list[i]
            img = add_serial_num(img, i)
            final_imgs_list[i] = img
        
        for im in final_imgs_list:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.width

        # 保存或显示拼接后的图像
        new_im.save(out_filename)


