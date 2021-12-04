import cv2
import numpy as np
from keras.preprocessing import image

# train_datagen = ImageDataGenerator(rescale = 1./255, horizontal_flip = False)
# training_set = train_datagen.flow_from_directory('/Users/anirudhnegi/Downloads/Section 40 - Convolutional Neural Networks (CNN)/dataset/train',
#                                                 target_size = (128, 128),
#                                                 batch_size = 32,
#                                                 class_mode = 'categorical')

dic = {0: 'no-right', 1: 'one-way', 2: 'do-not-enter', 3: 'block-ahead', 4: 'u-turn-permitted-ahead', 5: 'no-u-turn', 6: 'no-turn-on-red',
       7: 'left-lane-must', 8: 'no-pedestrian', 9: 'no-trucks', 10: 'no-left', 11: 'bus-only', 12: 'exit-only', 13: 'road-construction-ahead',
       14: 'pedestrian-crossing', 15: 'stop', 16: 'speed-limit-27', 17: 'speed-limit-18', 18: 'speed-limit-20', 19: 'only-two-way', 20: 'no-parking',
       21: 'right-or-through', 22: 'only-right', 23: 'speed-limit-30', 24: 'hospital', 25: 'bicycle-route', 26: 'watch-for-motorcycles',
       27: 'do-not-block', 28: 'one-way-right', 29: 'end-school-zone', 30: 'yield', 31: 'height-limit', 32: 'workers-ahead', 33: 'traffic-light',
       34: 'road-work-ahead', 35: 'one-way-left', 36: 'parking', 37: 'school', 38: 'speed-limit-25', 39: 'bus-stop', 40: 'no-access',
       41: 'speed-limit-40', 42: 'left-on-green-arrow-only', 43: 'slight-left', 44: 'bicycle-and-ped'}
# road_signs = dict([(value, key) for key, value in training_set.class_indices.items()])
road_signs = dict([(value, key) for key, value in dic.items()])


# json_file = open('../roadsigns.json', 'r')
# rs = json_file.read()
# json_file.close()
# loaded_model = model_from_json(rs)
# # load weights into new model
# loaded_model.load_weights("../roadsigns.h5")
# print("Loaded model from disk")


def class_predictor(loaded_model, img, coordinates):
    # for signs in coordinates:
    p1, p2 = coordinates[0], coordinates[1]
    sign_in_img = img[p1[1]:p2[1], p1[0]:p2[0]]
    sign_in_img = cv2.resize(sign_in_img, (128, 128))
    test_image = image.img_to_array(sign_in_img)
    test_image = np.expand_dims(test_image, axis=0)
    result = loaded_model.predict(test_image)
    font = cv2.FONT_HERSHEY_SIMPLEX

    fontScale = 1
    color = (255, 0, 0)
    thickness = 2

    for i, j in enumerate(result[0]):
        if (j == 1.0):
            img = cv2.putText(img, road_signs[i], (p1[0], p1[1]), font,
                              fontScale, color, thickness, cv2.LINE_AA)
            img = cv2.rectangle(img, p1, p2, (0, 255, 0), 2, 1)
            cv2.imwrite("frm.jpg", sign_in_img)
    return img
    # print("The object is: ", road_signs[i])
    # cv2.imshow("frm",img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
# timg=cv2.imread('test.jpg')
#
# class_predictor(timg,[(661,662),(716,756)])
