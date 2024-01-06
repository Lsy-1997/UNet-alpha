import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import cv2
from scipy.ndimage import maximum_filter
from utils.augmentations import Albumentations

def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
            
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        root_dir = os.path.split(images_dir)[0]
        classes_file = os.path.join(root_dir, 'classes.txt')
        if os.path.exists(classes_file):
            self.mask_values = []
            with open(classes_file, 'r') as file:
                for line in file:
                    item = int(line.strip())
                    self.mask_values.append(item)

        else:
            logging.info(f'Creating dataset with {len(self.ids)} examples')
            logging.info('Scanning mask files to determine unique values')
            with Pool() as p:
                unique = list(tqdm(
                    p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                    total=len(self.ids)
                ))

            self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
            logging.info(f'Unique mask values: {self.mask_values}')
            
            with open(classes_file, 'w') as file:
                for item in self.mask_values:
                    file.write(str(item) + '\n')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask, square_pad=False):
        
        # 将图像padding成正方形
        if square_pad == True:
            if pil_img.size[0]!=pil_img.size[1]:
                # 计算目标图像尺寸
                width, height = pil_img.size
                padding_size = max(width,height)

                # 创建新的图像对象，并填充为黑色
                new_image = Image.new(pil_img.mode, (padding_size, padding_size), (0, 0, 0) if is_mask==False else 0)
                
                # 将原始图像粘贴到新图像中心
                new_image.paste(pil_img, ((padding_size-width)//2, (padding_size-height)//2))
                
                pil_img = new_image
            
        # use fixed input size
        # pil_img = pil_img.resize((1024, 1024))
        img = np.asarray(pil_img)
        img = cv2.resize(img, (1024, 1024))

        # w, h = pil_img.size
        # newW, newH = int(scale * w), int(scale * h)
        # assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        # pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
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
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }

class TongjiParkingDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1, filter_size=3, use_intensity=0):
        super().__init__(images_dir, mask_dir, scale, filter_size)
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.filter_size = filter_size
        self.use_intensity = use_intensity
        self.augmentation = Albumentations()
            
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        root_dir = os.path.split(images_dir)[0]
        classes_file = os.path.join(root_dir, 'classes.txt')
        if os.path.exists(classes_file):
            self.mask_values = []
            with open(classes_file, 'r') as file:
                for line in file:
                    item = int(line.strip())
                    self.mask_values.append(item)

        else:
            logging.info(f'Creating dataset with {len(self.ids)} examples')
            logging.info('Scanning mask files to determine unique values')
            with Pool() as p:
                unique = list(tqdm(
                    p.imap(partial(unique_mask_values, mask_dir=self.mask_dir), self.ids),
                    total=len(self.ids)
                ))

            self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
            logging.info(f'Unique mask values: {self.mask_values}')
            
            with open(classes_file, 'w') as file:
                for item in self.mask_values:
                    file.write(str(item) + '\n')
                    
    def preprocess(self, mask_values, img, scale, is_mask, square_pad=False):

        if not is_mask:
            img = cv2.resize(img, (1024, 1024))
        else:
            img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        
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
            if self.use_intensity==1:
                rgb = img[:,:,0:3]
                intensity = img[:,:,3]
                filtered_intensity = maximum_filter(intensity, size=self.filter_size)
                filtered_intensity = filtered_intensity[...,None]
                img = np.concatenate((rgb, filtered_intensity),axis=2)
            elif self.use_intensity==2:
                intensity = img[:,:,3]
                filtered_intensity = maximum_filter(intensity, size=self.filter_size)
                filtered_intensity = filtered_intensity[...,None]
                img = filtered_intensity
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img
        
    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])
        
        img = np.asarray(img)
        mask = np.asarray(mask)
        
        if self.use_intensity == 1:
            img_rgb = img[:,:,0:3]
            img_intensity = img[:,:,3]
            img_intensity = img_intensity[..., None]

            img_rgb, mask = self.augmentation(img_rgb, mask)
            img = np.concatenate((img_rgb, img_intensity), axis=2)

        elif self.use_intensity == 0:
            img = img_rgb = img[:,:,0:3]
            img, mask = self.augmentation(img, mask)
            

        assert img.shape[0:1] == mask.shape[0:1], \
            f'Image and mask {name} should be the same size, but are {img.shape} and {mask.shape}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        
        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }

class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')
