import cv2


def load_img_color(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)


def load_img_grayscale(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def save_img(img, path):
    cv2.imwrite(path, img)


def load_all_imgs(dir_path):
    loaded_imgs = []
    with os.scandir(dir_path) as files:
        for filename in files:
            if filename.name.endswith('.jpg') and filename.is_file():
                loaded_imgs.append(load_img_color(filename.path))
    return loaded_imgs
