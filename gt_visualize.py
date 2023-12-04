import os 
from predict import plot_mask
from PIL import Image
import numpy as np
from tqdm import tqdm

imgs_dir = "../PSV_dataset/images/train"
labels_dir = "../PSV_dataset/labels/train"

imgs_file = os.listdir(imgs_dir)
imgs_basename = [os.path.basename(img_file)[0:-4] for img_file in imgs_file]

imgs_file_path = [os.path.join(imgs_dir, img_file) for img_file in imgs_file]

labels_file_path = [os.path.join(labels_dir,img_basename+'.png') for img_basename in imgs_basename]

save_dir = "./groudtruth_train"
if not os.path.exists(save_dir):
   os.makedirs(save_dir) 

for i in tqdm(range(len(imgs_file_path))):
    if os.path.basename(imgs_file_path[i])[0:-4] != os.path.basename(labels_file_path[i])[0:-4]:
        print("error")
        break
    img = Image.open(imgs_file_path[i])
    label = Image.open(labels_file_path[i])
    label = np.array(label)
    
    mask = np.zeros((6,label.shape[0],label.shape[1]), dtype=bool)
    for j in range(6):
        if j==0:
            continue
        mask[j][label==j] = True
    
    overlap_result = plot_mask(img=img, masks=mask, colors=[[0,0,0],[0,255,0],[255,0,0],[0,0,255],[255,255,0],[0,255,255]])
    
    output_path = os.path.join(save_dir, imgs_basename[i] + '.jpg')
    overlap_result.save(output_path)

