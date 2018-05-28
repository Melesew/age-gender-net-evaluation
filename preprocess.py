import cv2
import os
from scipy.io import loadmat

CURRENT_DIR = os.getcwd()
imdb_path = '../../datasets/imdb_crop'

def get_meta(mat_path, db):
    meta = loadmat(mat_path)
    full_path = meta[db][0, 0]["full_path"][0]
    gender = meta[db][0, 0]["gender"][0]

    return full_path, gender


full_path, gender = get_meta(imdb_path+"/imdb.mat", "imdb")

def load_data(mat_path):
    d = loadmat(mat_path)

    return d["image"], d["gender"][0]

def preprocess_for_gender_classifiaction():
    print (str(full_path.shape[0]) + " images found")

    countf = 0
    countm = 0
    countnan = 0

    #prepare folder for files
    female_path = CURRENT_DIR + '/gender_labled/F/'
    if not os.path.exists(female_path):
        female_path = os.mkdir(CURRENT_DIR + '/gender_labled/F/')

    male_path = CURRENT_DIR + '/gender_labled/M/'
    if not os.path.exists(male_path):
        male_path = os.mkdir(CURRENT_DIR + '/gender_labled/M/')

    for i in range(full_path.shape[0]):
        head, tail = os.path.split(full_path[i][0])

        # takes the first 21 folders from imdb_crop directory
        if head[0] == '0' or head[0] == '1' or (head[0] == '2' and head[1] == '0'):
            if gender[i] == 0.0 and countf <= 1000:
                newtail = female_path + tail
                _path = imdb_path +'/'+ str(full_path[i][0])

                t = cv2.imread(_path)
                tru = cv2.imwrite(newtail, t)
                if tru:
                    countf += 1

            elif gender[i] == 1.0 and countm <= 1000:
                newtail = male_path + tail
                _path = imdb_path +'/'+ str(full_path[i][0])
                t = cv2.imread(_path)
                tru = cv2.imwrite(newtail, t)

                if tru:
                    countm += 1
            else:
                countnan += 1

    print(countf, countm, countnan)

# No of files in a directory
def get_nb_files(directory):
    """Get number of files by searching directory recursively"""
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for files in os.walk(directory):
        cnt += len(files[2])
    return cnt

if __name__ == "__main__":

    preprocess_for_gender_classifiaction()

    # print(get_nb_files('M'))