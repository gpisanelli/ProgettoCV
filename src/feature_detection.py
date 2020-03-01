from cv2 import xfeatures2d


def detect_features_SIFT(img):
    sift = xfeatures2d.SIFT_create()
    return sift.detectAndCompute(img, mask=None)
