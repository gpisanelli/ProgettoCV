import cv2
import feature_detection
import time


def prova():
    t = cv2.imread('/home/mattia/Downloads/251722.jpg')
    print('PROVA')
    start = time.time()
    feature_detection.detect_features_SIFT(t)
    print(time.time() - start)
    print('-------------------------')


prova()