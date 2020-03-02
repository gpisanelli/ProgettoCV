from cv2 import xfeatures2d

sift = None

def detect_features_SIFT(img):
    global sift
    if sift is None:
        sift = xfeatures2d.SIFT_create()

    kp, des = sift.detectAndCompute(img, mask=None)
    return kp, des
