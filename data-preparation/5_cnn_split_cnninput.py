import os
# Directory Structure:
# dataset/train : contains images in classes directories (80%)
# dataset/test : contains images in classes directories (20%)

from shutil import copy
# dataset_collected is dataset directory containing images within subdirectories of each class
# dataset_collected/stop, dataset_collected/no-left, etc. which are cropped from video or the images by mapillary dataset
for dirn in os.listdir('dataset_collected'):
    if dirn.startswith('.'):
        continue
    # generate folder structure for training data for CNN model
    os.mkdir('dataset/train/'+dirn)
    os.mkdir('dataset/test/'+dirn)
    count=len(os.listdir('dataset_collected/'+dirn))
    # spliting training and test data in 80:20 ratio
    traintdata=int(0.8 * count)
    files=os.listdir('dataset_collected/'+dirn)
    for i in range(count):
        if(traintdata>0):
            copy( "dataset_collected/"+dirn+"/"+files[i], "dataset/train/"+dirn+"/"+files[i])
        else:
            copy( "dataset_collected/"+dirn+"/"+files[i], "dataset/test/"+dirn+"/"+files[i])
        traintdata -= 1