import cv2


def equalize_histogram(img):
    img_copy = img.copy()
    if len(img.shape) > 2:
        return img_copy

    cv2.equalizeHist(img_copy)
    return img_copy


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
