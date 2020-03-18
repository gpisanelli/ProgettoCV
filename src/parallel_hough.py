import copyreg
import math
import multiprocessing
import time
from contextlib import contextmanager
from functools import partial

import cv2
import numpy as np

import feature_detection
import image_processing
import prove
import visualization

low_H = 90
low_S = 8
low_V = 178
high_H = 129
high_S = 40
high_V = 210


def split_shelves(scene):
    frame_HSV = cv2.cvtColor(scene, cv2.COLOR_BGR2HSV)
    frame_threshold = cv2.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    frame_threshold = cv2.erode(frame_threshold, kernel=kernel, iterations=1)

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 20  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments

    lines_image = np.zeros(frame_threshold.shape, dtype=np.uint8)
    lines = cv2.HoughLinesP(frame_threshold, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    for line in lines:
        for x1, y1, x2, y2 in line:
            angle = math.atan2(y1 - y2, x1 - x2)
            if math.radians(-2) <= angle <= math.radians(2) or math.radians(178) <= angle <= math.radians(182):
                cv2.line(lines_image,
                         (x1 - int(1000 * math.cos(angle)), y1 - int(1000 * math.sin(angle))),
                         (x2 + int(1000 * math.cos(angle)), y2 + int(1000 * math.sin(angle))), 255, 1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    lines_image = cv2.morphologyEx(lines_image, cv2.MORPH_DILATE, kernel=kernel, iterations=15)
    lines_image = cv2.morphologyEx(lines_image, cv2.MORPH_ERODE, kernel=kernel, iterations=15)
    lines_image = cv2.Canny(lines_image, 200, 255)

    rows = []
    mean_lines = []
    for row in range(lines_image.shape[0]):
        if lines_image[row, 100] > 0:
            rows.append(row)

    if len(rows) % 2 != 0:
        rows.append(scene.shape[0] - 1)

    i = 0
    while i < len(rows):
        mean_lines.append(int((rows[i] + rows[i + 1]) / 2))
        i += 2

    mean_lines_img = np.zeros(scene.shape, dtype=np.uint8)
    for row in mean_lines:
        cv2.line(mean_lines_img, (0, row), (mean_lines_img.shape[1], row), (255, 0, 0), 1)

    visualization.display_img(mean_lines_img)

    sub_images = []
    sub_images.append((scene[0:mean_lines[0], 0:scene.shape[1]], 0))
    for i in range(1, len(mean_lines)):
        if mean_lines[i] - mean_lines[i-1] >= scene.shape[0] // 10:
            sub_images.append((scene[mean_lines[i-1]:mean_lines[i], 0:scene.shape[1]], mean_lines[i-1]))

    if scene.shape[0] - mean_lines[len(mean_lines)-1] >= scene.shape[0] // 10:
        sub_images.append((scene[mean_lines[len(mean_lines)-1]:scene.shape[0], 0:scene.shape[1]], mean_lines[len(mean_lines)-2]))

    return sub_images


def preprocess_sub_scene(s):
    # Preprocessing (scene image)
    pr_scene = s.copy()
    pr_scene = image_processing.convert_grayscale(pr_scene)
    pr_scene = image_processing.equalize_histogram(pr_scene)
    pr_scene = image_processing.sharpen_img(pr_scene)

    return pr_scene


def _pickle_keypoints(point):
    return cv2.KeyPoint, (*point.pt, point.size, point.angle,
                          point.response, point.octave, point.class_id)


def compute_sub_image(dict_box_features, sub_image):
    sub_scene, y = sub_image
    proc_scene = preprocess_sub_scene(sub_scene)
    test_scene = image_processing.resize_img_dim(sub_scene, proc_scene.shape[1], proc_scene.shape[0])

    s = time.time()
    kp_scene, des_scene = feature_detection.detect_features_SIFT(proc_scene)
    #print('\nTIME DETECTION: ', time.time() - s, '\n')
    found_bounds = {}
    for box_name in dict_box_features:
        found_bounds[box_name] = []

        # Box features retrieval
        box, proc_box, kp_box, des_box = dict_box_features[box_name]

        bounds = prove.compute_hough(box, kp_box, des_box, test_scene, kp_scene, des_scene, box, sub_scene)
        for b in bounds:
            shifted_bound = b.copy()
            for i in range(len(b)):
                shifted_bound[i] = [b[i][0], b[i][1]+y]
            found_bounds[box_name].append(shifted_bound)

    return found_bounds


@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


def hough_sub_images(sub_images, dict_template_features):
    copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoints)
    with poolcontext(processes=len(sub_images)) as pool:
        results = pool.map(partial(compute_sub_image, dict_template_features), sub_images)

    return results
