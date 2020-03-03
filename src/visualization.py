import cv2
import numpy as np
import image_processing
import platform
import subprocess
from win32api import GetMonitorInfo, MonitorFromPoint

if platform.system() == 'Windows':
    screen_width = GetMonitorInfo(MonitorFromPoint((0, 0))).get("Work")[2]
    screen_height = GetMonitorInfo(MonitorFromPoint((0, 0))).get("Work")[3] - 32   # subtracting title bar height
else:
    output = subprocess.Popen('xrandr | grep "\*" | cut -d" " -f4', shell=True, stdout=subprocess.PIPE).communicate()[0]
    resolution = output.split()[0].split(b'x')
    screen_width = resolution[0]
    screen_height = resolution[1]


def display_img(img, width=0, title='Image'):
    if width != 0:
        img = image_processing.resize_img_width(img, width)

    img = correct_if_oversized(img)

    cv2.imshow(title, img)
    while True:
        key = cv2.waitKey(10)

        # Wait for ESC key
        if key == 27:
            break

    cv2.destroyAllWindows()


def correct_if_oversized(img):
    oversized = oversized_bool_list(img)
    while oversized[0] or oversized[1]:
        if oversized[0]:
            img = image_processing.resize_img_height(img, screen_height)
        else:
            img = image_processing.resize_img_width(img, screen_width)
        oversized = oversized_bool_list(img)
    return img


def oversized_bool_list(img):
    return [img.shape[0] > screen_height, img.shape[1] > screen_width]


def draw_polygons(image, polygons):
    img_copy = image.copy()
    for polygon in polygons:
        cv2.polylines(img_copy, [polygon], True, (0, 255, 0), 5)

    return img_copy
