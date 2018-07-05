
# AGE and GENDER recognition Tensorflow DNN

Prerequestes
Install this to Run Projects on Your Machine

    Tensorflow 1.5
    Keras

# model evaluations

source = [IMDB-WIKI – 500k+ face images with age and gender labels](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)

main.py train the network and saves the model, gen.hs

extract_imdb_mat.py takes the first 21 (00, 01, 02, ..., 20) folders from imdb_crop directory and extract labeled data.

facecrop.py takes the labeled datas and crop faces and put them in cropped directory

predict.py takes cropped images, predict thier gedner and move them on their respective directory

visually_inspect.py visualizes the models performance


**to test with certain pics** = python3 predict.py --model "models/gen.h5" --image "path to image "

**to test with web-cam** = python demo_from_cam.py


