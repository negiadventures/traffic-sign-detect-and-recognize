import shutil

import cv2
import os
train = 'cnn_train/'
c=0
for d in os.listdir(train):
    # if number of images in a class is very few, remove the class
    if len(os.listdir(train + d)) <20:
        shutil.rmtree(train + d)
        c+=1
print(c)

# TODO: instead of removing, we can do data augmentation since in real life these signs will apear on roads.


# # get dimensions of image
#
# # height, width, number of channels in image
# height = img.shape[0]
# width = img.shape[1]
# channels = img.shape[2]
#
# print('Image Dimension    : ', dimensions)
# print('Image Height       : ', height)
# print('Image Width        : ', width)
# print('Number of Channels : ', channels)
