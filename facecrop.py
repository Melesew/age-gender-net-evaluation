import os
import csv
import numpy as np
import random
import cv2

DATA_PATH = 'gender_labled'

CURRENT_DIR = os.getcwd()

# Function used to crop an image and place in a folder
def facecrop(image, cropped_dir):
    # @param image is the image path plus its name
    facedata = "haarcascade_frontalface_alt.xml"
    cascade = cv2.CascadeClassifier(facedata)

    img = cv2.imread(image)

    minisize = (img.shape[1], img.shape[0]) # [height, width]
    miniframe = cv2.resize(img, minisize)

    faces = cascade.detectMultiScale(miniframe)

    for f in faces:
        x, y, w, h = [v for v in f]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255))

        sub_face = img[y:y + h, x:x + w]
        img_name = image.split('/')[-1]
        fname, ext = os.path.splitext(img_name)
        cv2.imwrite(cropped_dir+'/' + fname + ext, sub_face)

    return

# Function to iterate over images directory and call the facecrop fun. to folder croped.
def read_through(dirname):
    count = 0

    # Prepare directory proccessed images
    cropped_dir = CURRENT_DIR + '/cropped'
    if not os.path.exists(cropped_dir):
        cropped_dir = os.mkdir(CURRENT_DIR + '/cropped')

    for cur, _dirs, files in os.walk(dirname):
        head, tail = os.path.split(cur)
        while head:
            head, _tail = os.path.split(head)

        for f in files:
            if ".jpg" in f:
                image = dirname + "/" + tail + "/" + f
                # print (f, file_path)
                facecrop(image, cropped_dir)

                count += 1
        print(count)

if __name__ == "__main__":
    read_through(DATA_PATH)