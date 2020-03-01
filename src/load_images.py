import cv2


def load_img_color(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)


def load_img_grayscale(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def save_img(img, path):
    cv2.imwrite(path, img)
