import cv2


def detect_features_SIFT(img):
    sift = cv2.xfeatures2d.SIFT_create()
    return sift.detectAndCompute(img, mask=None)
