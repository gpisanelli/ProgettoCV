import cv2
import numpy as np


def convert_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def equalize_histogram(img):
    if len(img.shape) > 2:
        return img

    cv2.equalizeHist(img)
    return img


def sharpen_img(img):
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(img, 1.5, blurred, -0.5, 0)


def denoise_img(img):
    return cv2.fastNlMeansDenoising(img, None, 10, 7, 21)


def resize_img(img, scale_factor):
    width = int(img.shape[1] * scale_factor)
    height = int(img.shape[0] * scale_factor)
    return cv2.resize(img, (width,height), interpolation=cv2.INTER_CUBIC)


def resize_img_dim(img, width, height):
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)


def resize_img_width(img, width):
    height = int(img.shape[0] * width/img.shape[1])
    return cv2.resize(img, (width,height), interpolation=cv2.INTER_CUBIC)


def resize_img_height(img, height):
    width = int(img.shape[1] * height/img.shape[0])
    return cv2.resize(img, (width,height), interpolation=cv2.INTER_CUBIC)


def transform_box_in_scene(box, scene, homography):
    transformed_box = cv2.warpPerspective(box, homography, (scene.shape[1], scene.shape[0]))
    test_scene = scene.copy()
    mask = cv2.split(transformed_box)[0].astype(dtype=np.int32) + \
           cv2.split(transformed_box)[1].astype(dtype=np.int32) + \
           cv2.split(transformed_box)[2].astype(dtype=np.int32)

    mask[mask > 0] = 1
    test_scene[:, :, 0] = test_scene[:, :, 0] * mask
    test_scene[:, :, 1] = test_scene[:, :, 1] * mask
    test_scene[:, :, 2] = test_scene[:, :, 2] * mask

    return transformed_box, test_scene
