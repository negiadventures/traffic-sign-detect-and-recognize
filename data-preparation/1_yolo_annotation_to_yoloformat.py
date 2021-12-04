import json
import os

import pandas as pd

# Data collected from: https://www.mapillary.com/dataset/trafficsign
# yolo label file format: 
#class (center of x cooardinate)/width_of_image (center of y cooardinate)/height_of_image width_of_object/width_of_image   height_of_object/height_of_image

# i.e.   0 ((xmax+xmin)/2)/width ((ymax+ymin)/2)/height  (xmax-xmin)/width (ymax-ymin)/height

directory = 'mtsd_fully_annotated_annotation/mtsd_v2_fully_annotated/annotations/'
yolo_inp_data = 'annotations/'
for file in os.listdir(directory):
    if file.endswith(".json"):
        # converting annotation .json files to required yolo format
        with open(directory + file, "r") as read_file:
            data = json.load(read_file)
            df = pd.DataFrame(columns=['o', 'x', 'y', 'width', 'height'])
            for obj in data['objects']:
                if (obj['bbox']['xmax'] - obj['bbox']['xmin']) > 40 and (obj['bbox']['ymax'] - obj['bbox']['ymin']) > 40 and (
                        obj['label'] != 'other-sign'):
                    df.loc[len(df)] = ['0', (obj['bbox']['xmax'] + obj['bbox']['xmin']) / (2 * data['width']),
                                       (obj['bbox']['ymax'] + obj['bbox']['ymin']) / (2 * data['height']),
                                       (obj['bbox']['xmax'] - obj['bbox']['xmin']) / data['width'],
                                       (obj['bbox']['ymax'] - obj['bbox']['ymin']) / data['height']]
            df.to_csv(yolo_inp_data + file.split('/')[-1:][0].replace('.json', '.txt'), sep=' ', header=False, index=False)

# https://towardsdatascience.com/image-data-labelling-and-annotation-everything-you-need-to-know-86ede6c684b1
