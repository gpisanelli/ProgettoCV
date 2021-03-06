import copyreg
import math
import multiprocessing
from contextlib import contextmanager
from functools import partial

import cv2
import numpy as np

import feature_matching
import hough_detection
import object_validation
from utils import visualization, image_processing

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

    sub_images = []
    sub_images.append((scene[0:mean_lines[0], 0:scene.shape[1]], 0))
    for i in range(1, len(mean_lines)):
        if mean_lines[i] - mean_lines[i-1] >= scene.shape[0] // 10:
            sub_images.append((scene[mean_lines[i-1]:mean_lines[i], 0:scene.shape[1]], mean_lines[i-1]))

    if scene.shape[0] - mean_lines[len(mean_lines)-1] >= scene.shape[0] // 10:
        sub_images.append((scene[mean_lines[len(mean_lines)-1]:scene.shape[0], 0:scene.shape[1]], mean_lines[len(mean_lines)-1]))

    return sub_images


def preprocess_sub_scene(s):
    pr_scene = s.copy()
    pr_scene = image_processing.convert_grayscale(pr_scene)
    return pr_scene


def erase_bar(matches_mask, bounds):
    # per cancellare dalla mask il baricentro disegno sopra un baricentro nero delle stesse dimensioni
    M_intersected = cv2.moments(bounds)
    cx_intersected = int(M_intersected['m10'] / M_intersected['m00'])
    cy_intersected = int(M_intersected['m01'] / M_intersected['m00'])
    _, _, w_intersected, _ = cv2.boundingRect(bounds)
    cv2.circle(matches_mask, (cx_intersected, cy_intersected), w_intersected // 4, (0, 0, 0), -1)


def compute_sub_image(dict_box_features, scaling_factor, sub_image):
    sub_scene, y = sub_image
    sub_scene = image_processing.resize_img(sub_scene, scaling_factor)
    proc_scene = preprocess_sub_scene(sub_scene)
    y *= scaling_factor
    test_scene = image_processing.resize_img_dim(sub_scene, proc_scene.shape[1], proc_scene.shape[0])

    kp_scene, des_scene = feature_matching.detect_features_SIFT(proc_scene)

    found_bounds = {}

    matches_mask = np.zeros(proc_scene.shape, dtype=np.uint8)
    color = 1
    bounds_dict = {}

    for box_name in dict_box_features:
        found_bounds[box_name] = []

        box, proc_box, kp_box, des_box = dict_box_features[box_name]

        matches = feature_matching.find_matches(des_box, des_scene, dist_threshold=0.85)
        good_matches = hough_detection.find_matches(matches, kp_box, kp_scene, proc_scene)

        result_bounds = []
        for i in good_matches:
            if len(good_matches[i][1]) >= 4:
                success, bounds, homography, used_src_pts, used_dst_pts, not_used_matches = \
                    feature_matching.find_object(good_matches[i][1], kp_box, kp_scene, proc_box)

                if success and \
                        object_validation.is_convex_polygon(bounds) and \
                        object_validation.validate_color(box, test_scene, used_src_pts, used_dst_pts, bounds, homography, remove_feature_areas=False, hue_threshold=0.7) and \
                        object_validation.check_rectangularity(bounds, threshold=0.8):

                    result_bounds.append(bounds)

        for b in result_bounds:
            M = cv2.moments(b)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            _, _, w, _ = cv2.boundingRect(b)
            new_barycenter_mask = np.zeros(matches_mask.shape, dtype=np.uint8)
            cv2.circle(new_barycenter_mask, (cx, cy), w // 4, color, -1)
            new_bar_copy = new_barycenter_mask.copy()
            new_bar_copy[new_bar_copy > 0] = 255
            intersection = cv2.bitwise_and(new_bar_copy, matches_mask)

            if cv2.countNonZero(intersection) > 0:
                gray_index = intersection[intersection > 0][0]
                bar_intersected = bounds_dict[gray_index]
                bar_intersecting = (b, box, box_name)
                result = object_validation.compare_detections(bar_intersected, bar_intersecting)
                #if result == 1 allora l'intersecato è interno all'intersecante, non faccio nulla
                if result == 0:  # devo sostituire nel dict e nella mask l'intersecato con l'intersecante
                    erase_bar(matches_mask, bar_intersected[0])
                    cv2.circle(matches_mask, (cx, cy), w // 4, int(gray_index), -1)
                    bounds_dict[gray_index] = bar_intersecting
                if result == -1:  # c'è intersezione ma i box non sono completamente uno dentro l'altro, effettuo ulteriore controllo con compare_detections_hard
                    #print('Calling compare_detections_hard')
                    result = object_validation.compare_detections_hard(bar_intersected, bar_intersecting, 2)
                    #print('Result of compare_detections_hard = ', result)
                    # if result == 1 allora c'è troppa sovrapposizione tra i box e l'intersecato è migliore dell'intersecante, non faccio nulla
                    if result == 0:  # c'è troppa sovrapposizione tra i box e l'intersecante è migliore dell'intersecato, sostituisco
                        erase_bar(matches_mask, bar_intersected[0])
                        cv2.circle(matches_mask, (cx, cy), w // 4, int(gray_index), -1)
                        bounds_dict[gray_index] = bar_intersecting
                    if result == -1:  # anche secondo compare_detections_hard c'è intersezione ma non sovrapposizione, aggiungo al dict e alla mask
                        cv2.circle(matches_mask, (cx, cy), w // 4, color, -1)
                        bounds_dict[color] = bar_intersecting
                        color += 1
            else:  # niente intersezione, aggiungo al dict e alla mask
                cv2.circle(matches_mask, (cx, cy), w // 4, color, -1)
                bounds_dict[color] = (b, box, box_name)
                color += 1

    for key in bounds_dict:
        bound, box, box_name = bounds_dict[key]
        shifted_bound = bound.copy()
        for i in range(len(bound)):
            shifted_bound[i] = [bound[i][0], bound[i][1] + y]
        found_bounds[box_name].append(shifted_bound)

    return found_bounds


def _pickle_keypoints(point):
    return cv2.KeyPoint, (*point.pt, point.size, point.angle,
                          point.response, point.octave, point.class_id)


@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


def hough_sub_images(scene, dict_template_features, scaling_factor):
    sub_scenes = split_shelves(scene)

    copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoints)
    with poolcontext(processes=len(sub_scenes)) as pool:
        results = pool.map(partial(compute_sub_image, dict_template_features, scaling_factor), sub_scenes)

    return results