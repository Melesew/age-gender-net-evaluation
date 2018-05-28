import numpy as np
import os
import glob
import cv2

import matplotlib.pyplot as plt

classes = ['female', 'male']

predicted_male_images = glob.glob(os.path.join('predicted', 'male', '*.jpg'))

predicted_female_images = glob.glob(os.path.join('predicted', 'female', '*.jpg'))

# print(len(female_predicted_images), len(male_predicted_images))

true_male_images = glob.glob(os.path.join('gender_labled', 'M', '*.jpg'))
true_female_images = glob.glob(os.path.join('gender_labled', 'F', '*.jpg'))

def get_image_names(file):
    image_name = []
    for f in file:
        image_name.append(f.split('/')[-1])
    return image_name

predicted_male_image_names = get_image_names(predicted_male_images)
true_male_image_names = get_image_names(true_male_images)

predicted_female_image_names = get_image_names(predicted_female_images)
true_female_image_names = get_image_names(true_female_images)

correct_countf = 0
correct_countm = 0
incorrect_count = 0

incorrectly_predicted_female = []

#Correctly predicted females
for img_name in predicted_female_image_names:
    if img_name in true_female_image_names:
        correct_countf += 1
    else:
        incorrect_count +=1
        # incorrectly_predicted_female.append(img_name)

incorrectly_predicted_male = []

for img_name in predicted_male_image_names:
    if img_name in true_male_image_names:
        correct_countm += 1
    else:
        incorrect_count +=1
        # incorrectly_predicted_male.append(img_name)

accuracy = (correct_countf + correct_countm) / (len(predicted_female_images) + len(predicted_male_images))

print('accuracy : ', accuracy)


