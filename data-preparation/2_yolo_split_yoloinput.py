import os
import random
from shutil import copy

# Directory Structure:
# dataset/images/train : contains images (80%)
# dataset/images/valid : contains images (20%)
# dataset/labels/train : contains labels (80%)
# dataset/labels/valid : contains labels (20%)

# signs.yaml : train, val location, number of classes and names of classes

org_images_path = 'mtsd_fully_annotated_images.train.0/'
ann = 'annotations/'
yolo_inp_images = 'dataset/'
img_train = yolo_inp_images + 'images/train'
# img_test  = path + '/images/test'
img_valid = yolo_inp_images + 'images/valid'
ann_train = yolo_inp_images + 'labels/train'
# ann_test  = path + '/labels/test'
ann_valid = yolo_inp_images + 'labels/valid'
# splitting dataset into 80:20 ratio
train = 0.8
validate = 0.2
# test = 0
for i, filename in enumerate(os.listdir(org_images_path)):
    img_src = org_images_path + filename
    ann_src = ann + filename.split('.')[0] + '.txt'
    if os.path.isfile(ann_src) and os.path.isfile(img_src):
        rand = random.random()
        if rand <= train:
            img_dst = img_train
            ann_dst = ann_train
        elif rand <= (train + validate):
            img_dst = img_valid
            ann_dst = ann_valid
        copy(src=img_src, dst=img_dst)
        copy(src=ann_src, dst=ann_dst)
