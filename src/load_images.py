import cv2
import os.path


def get_path_for_box(box_name):
    return os.path.join('..', 'images', 'object_detection_project', 'models', box_name)


def get_path_for_scene(scene_name):
    return os.path.join('..', 'images', 'object_detection_project', 'scenes', scene_name)


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
