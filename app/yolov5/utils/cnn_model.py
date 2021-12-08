import cv2
import numpy as np
from keras.preprocessing import image

# train_datagen = ImageDataGenerator(rescale = 1./255, horizontal_flip = False)
# training_set = train_datagen.flow_from_directory('/Users/anirudhnegi/Downloads/Section 40 - Convolutional Neural Networks (CNN)/dataset/train',
#                                                 target_size = (128, 128),
#                                                 batch_size = 32,
#                                                 class_mode = 'categorical')

classes = [('bicycle-and-ped', 0), ('bicycle-route', 1), ('block-ahead', 2), ('bus-only', 3), ('bus-stop', 4), ('do-not-block', 5),
           ('do-not-enter', 6), ('end-school-zone', 7), ('exit-only', 8), ('height-limit', 9), ('hospital', 10), ('left-lane-must', 11),
           ('left-on-green-arrow-only', 12), ('no-access', 13), ('no-left', 14), ('no-parking', 15), ('no-pedestrian', 16), ('no-right', 17),
           ('no-trucks', 18), ('no-turn-on-red', 19), ('no-u-turn', 20), ('one-way', 21), ('one-way-left', 22), ('one-way-right', 23),
           ('only-right', 24), ('only-two-way', 25), ('parking', 26), ('pedestrian-crossing', 27), ('right-or-through', 28),
           ('road-construction-ahead', 29), ('road-work-ahead', 30), ('school', 31), ('slight-left', 32), ('speed-limit-18', 33),
           ('speed-limit-20', 34), ('speed-limit-25', 35), ('speed-limit-27', 36), ('speed-limit-30', 37), ('speed-limit-40', 38), ('stop', 39),
           ('traffic-light', 40), ('u-turn-permitted-ahead', 41), ('watch-for-motorcycles', 42), ('workers-ahead', 43), ('yield', 44)]

# road_signs = dict([(value, key) for key, value in training_set.class_indices.items()])
road_signs = dict([(value, key) for key, value in classes])

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
    sign = ''
    for i, j in enumerate(result[0]):
        if (j == 1.0):
            sign = road_signs[i]
            img = cv2.putText(img, sign, (p1[0], p1[1]), font,
                              fontScale, color, thickness, cv2.LINE_AA)
            img = cv2.rectangle(img, p1, p2, (0, 255, 0), 2, 1)
    return img, sign
    # print("The object is: ", road_signs[i])
    # cv2.imshow("frm",img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
# timg=cv2.imread('test.jpg')
#
# class_predictor(timg,[(661,662),(716,756)])
