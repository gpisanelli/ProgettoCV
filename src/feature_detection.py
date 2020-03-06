import multiprocessing
import numpy as np
import copyreg
from cv2 import xfeatures2d
import cv2


def compute_sift(sub_img):
    sift = xfeatures2d.SIFT_create(5000)
    kp_t, des_t = sift.detectAndCompute(sub_img, None)
    return kp_t, des_t


def _pickle_keypoints(point):
    return cv2.KeyPoint, (*point.pt, point.size, point.angle,
                          point.response, point.octave, point.class_id)


# DON'T USE: WORSE PERFORMANCE THAN NON PARALLEL
# Parallel feature detection that splits the image and computes the keypoint and descriptors of each chunk in a
# different thread. The final partial results are put together after all threads have completed their execution.
def detect_features_SIFT_parallel(img):
    arr_number = 8
    sub_imgs = []
    coords = []
    rows = 2
    cols = 4
    for i in range(0, rows):
        for j in range(0, cols):
            # Round to ceiling
            x_offset = img.shape[1] / cols / 8
            y_offset = img.shape[0] / rows / 8
            if j == arr_number - cols:
                x_offset = 0
            if i == arr_number - cols:
                y_offset = 0
            coords.append((int(j*img.shape[1]/cols), int(i*img.shape[0]/rows)))
            sub_imgs.append(img[int(i*img.shape[0]/rows):
                                int((i+1)*img.shape[0]/rows+y_offset),
                                int(j*img.shape[1]/cols):
                                int((j+1)*img.shape[1]/cols+x_offset)])

    copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoints)
    pool = multiprocessing.Pool(arr_number)
    results = pool.map(compute_sift, sub_imgs)

    kp, des = results[0]
    for i in range(1, arr_number):
        kp_i, des_i = results[i]
        for index, point in np.ndenumerate(kp_i):
            kp_i[index[0]].pt = (point.pt[0] + coords[i][0], point.pt[1] + coords[i][1])
        kp = kp + kp_i
        des = np.concatenate((des, des_i))
    
    sift = xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    
    return kp, des


def detect_features_SIFT(img):
    return compute_sift(img)
