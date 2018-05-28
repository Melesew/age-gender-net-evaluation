# model evaluations

source = [IMDB-WIKI – 500k+ face images with age and gender labels](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)

preprocess.py takes the first 21 (00, 01, 02, ..., 20) folders from imdb_crop directory and generate
    labeled datas.

facecrop.py takes the labeled datas and crop faces and put them merged on cropped directory

predict.py takes cropped images, predict thier gedner and move them on their respective directory

visually_inspect.py visualizes the models performance

