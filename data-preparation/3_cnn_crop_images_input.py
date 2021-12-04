import cv2
import os
import json
import pandas as pd
import math
import time

# image path for one of the downloaded images zip from mapillary dataset
images_path = 'mtsd_fully_annotated_images.train.0/'
# annotation path for downloaded annotation zip from mapillary dataset
ann = 'annotations'
print(time.time())
for i, filename in enumerate(os.listdir(images_path)):
    ann_fn = filename.replace('.jpg','.json')
    img = cv2.imread(images_path + filename)
    with open(ann + '/' + ann_fn, "r") as read_file:
        data = json.load(read_file)
        for obj in data['objects']:
            xmax = math.ceil(obj['bbox']['xmax'])
            xmin = math.floor(obj['bbox']['xmin'])
            ymax = math.ceil(obj['bbox']['ymax'])
            ymin = math.floor(obj['bbox']['ymin'])
            if (xmax - xmin) < 40 or (ymax - ymin) < 40:
                continue
            dir = obj['label']
            if len(obj['label'].split('--')) > 2:
                dir = '--'.join(obj['label'].split('--')[1:-1])
            # cropping and collecting sign board images using coordinates and classes from annotation files.
            crop_img = img[ymin:ymax, xmin:xmax]
            if not os.path.exists('cnn_train/' + dir):
                os.mkdir('cnn_train/' + dir)
            sign_name = len(os.listdir('cnn_train/' + dir)) + 1
            if len(crop_img)>0:
                try:
                    cv2.imwrite('cnn_train/' + dir + '/' + str(sign_name)+'.jpg', crop_img)
                except:
                    pass
            # cv2.imshow("cropped", crop_img)
            # cv2.waitKey(0)
print(time.time())