import os
import cv2

dataset_dir = './data/PSV_dataset'

images_dir = os.path.join(dataset_dir, 'images')
labels_dir = os.path.join(dataset_dir, 'labels')

images_list = os.listdir(images_dir)
labels_list = os.listdir(labels_dir)
images_path = []
for image in images_list:
    images_path.append(os.path.join(dataset_dir, os.listdir(image)))


labels_path = []
for image in labels_list:
    labels_path.append(os.path.join(dataset_dir, os.listdir(labels_dir)))


for image_path in images_path:
    cv2.imread(image_path)
    image_new_path
    cv2.imwrite()

