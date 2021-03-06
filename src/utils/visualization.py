import time

import cv2 as cv2
import numpy as np
import platform
import subprocess
from utils import load_images, image_processing

if platform.system() == 'Windows':
    from win32api import GetMonitorInfo, MonitorFromPoint
    screen_width = GetMonitorInfo(MonitorFromPoint((0, 0))).get("Work")[2]
    screen_height = GetMonitorInfo(MonitorFromPoint((0, 0))).get("Work")[3] - 32   # subtracting title bar height
else:
    output = subprocess.Popen('xrandr | grep "\*" | cut -d" " -f4', shell=True, stdout=subprocess.PIPE).communicate()[0]
    resolution = output.split()[0].split(b'x')
    screen_width = int(resolution[0])
    screen_height = int(resolution[1]) - 200


def display_img(img, width=0, title=None, wait=True):
    if title is None:
        title = str(time.time())
    if width != 0:
        img = image_processing.resize_img_width(img, width)

    img = correct_if_oversized(img)

    cv2.imshow(title, img)
    while True:
        key = cv2.waitKey(10)

        # Wait for ESC key
        if key == 27:
            break
    if wait:
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
        x, y, w, h = cv2.boundingRect(polygon)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 5)
        #cv2.polylines(img_copy, [polygon], True, (0, 255, 0), 5)

    return img_copy


def draw_bounding_rect(image, rect):
    x, y, w, h = rect
    img_copy = image.copy()
    cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 4)

    return img_copy


def draw_names(img, rect, box_name):
    rec_mask = np.zeros(img.shape, np.uint8)
    x, y, w, h = rect

    name = load_images.get_box_name(box_name)
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.7
    thickness = 2
    rec_mask = cv2.rectangle(rec_mask,
                             (x - 5, int(y + h/2 - 35*fontScale)),
                             (int(x + len(name)*20*fontScale), int(y + h/2 + 15*fontScale)),
                             (255, 255, 255),
                             thickness=-1)
    img = cv2.addWeighted(img, 1, rec_mask, 0.5, 0)
    img = cv2.putText(img, name, (x + 5, int(y + h/2)), fontFace, fontScale, (0, 0, 0), thickness)

    return img
