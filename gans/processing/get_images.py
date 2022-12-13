# helper function for selecting 'size' real images
# and downscaling them to lower dimension SPATIAL_DIM
import cv2
import numpy as np
from gans.utils.params import SPATIAL_DIM, LOCAL_DATA_PATH

def get_real_celebrity(df, size, total):
    cur_files = df.sample(frac=1).iloc[0:size]
    X = np.empty(shape=(size, SPATIAL_DIM, SPATIAL_DIM, 3))
    for i in range(0, size):
        file = cur_files.iloc[i]
        img_uri = LOCAL_DATA_PATH + 'img_align_celeba/' + file.image_id
        img = cv2.imread(img_uri)
        img = cv2.resize(img, (SPATIAL_DIM, SPATIAL_DIM))
        img = np.flip(img, axis=2)
        img = img.astype(np.float32) / 127.5 - 1.0
        X[i] = img
    return X
