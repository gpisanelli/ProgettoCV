from cv2 import xfeatures2d


def detect_features_SIFT(img):
    sift = xfeatures2d.SIFT_create()

    kp, des = sift.detectAndCompute(img, mask=None)
    return kp, des
