import sys
import argparse
import numpy as np
from PIL import Image

# from facecrop import facecrop
import os
import cv2
import shutil
import glob

from keras.preprocessing import image
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input

target_size = (64, 64)
DATA_PATH = 'cropped'

CURRENT_DIR = os.getcwd()

def predict(model, img, target_size):
    if img.size != target_size:
        img = img.resize(target_size)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)

    return preds[0]

def get_classNames(input):
    '''
    :param input: tuple or string representing age
    :return: number representing one of the age classes
    '''
    if input == 0:
        return "female"
    elif input == 1:
        return "male"


#Put images to their respective folder based on predicted value
def folder_files(image_path, predicted_dir):
    '''
    :param image_path: path to images
    :param predicted_path: destination path of images to their respective dir.(male/female)
    :return: number representing one of the age classes
    '''
    model = load_model('models/gen.h5')

    #Prepare path to predicted images

    male_predicted_dir = predicted_dir +'/male/'
    if not os.path.exists(male_predicted_dir):
        male_predicted_dir = os.mkdir(predicted_dir +'/male/')

    female_predicted_dir = predicted_dir + '/female/'
    if not os.path.exists(female_predicted_dir):
        female_predicted_dir = os.mkdir(predicted_dir + '/female/')

    path = os.path.join(image_path, '*.jpg')
    files = glob.glob(path)
    for fl in files:  # 774 files
        img = Image.open(fl)
        preds = predict(model, img, target_size)

        class_num = np.argmax(preds)
        # print(fl)
        # file_name = fl.split('/')[-1]
        if (get_classNames(class_num) == 'female'):
            shutil.move(fl, female_predicted_dir)

        else:
            shutil.move(fl, male_predicted_dir)

        # print(fl, get_classNames(class_num))

if __name__ == "__main__":
    # predicted_dir = CURRENT_DIR + '/predicted'
    # if not os.path.exists(predicted_dir):
    #     predicted_dir = os.mkdir(CURRENT_DIR + '/predicted')
    #
    # folder_files(DATA_PATH, predicted_dir)

    a = argparse.ArgumentParser()
    a.add_argument("--image", help="path to image")
    a.add_argument("--image_array")
    a.add_argument("--model")
    args = a.parse_args()

    if args.image is None:
        a.print_help()
        sys.exit(1)

    model = load_model(args.model)
    if args.image is not None:
        img = Image.open(args.image)
        preds = predict(model, img, target_size)
        print(np.argmax(preds))
    else:
        print ("bzw")