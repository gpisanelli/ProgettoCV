import math
import time

import cv2
import numpy as np

import feature_detection
import feature_matching
import image_processing
import load_images
import visualization


# (cos(θ) −sin(θ))    (x)
# (sin(θ) cos(θ) ) *  (y)
# theta: angle in radians
# vector: numpy array
def rotate(vector, theta):
    # Create rotation matrix
    rot_matrix = np.array(((np.cos(theta), -np.sin(theta)),
                           (np.sin(theta), np.cos(theta))))

    # Apply the rotation matrix to the vector
    result = rot_matrix.dot(vector)
    return result


def compute_accumulator(joining_vectors, sceneImg_shape, kp_scene):
    accumulator = np.zeros(sceneImg_shape)
    for i in joining_vectors:
        accum_i = joining_vectors[i][1][0] + kp_scene[i].pt[0]
        accum_j = joining_vectors[i][1][1] + kp_scene[i].pt[1]

        if 0 <= accum_i < accumulator.shape[1] and 0 <= accum_j < accumulator.shape[0]:
            accumulator[int(round(accum_j)), int(round(accum_i))] += 1

    return accumulator


def compute_joining_vectors_table(m, b, box_keypoints, scene_keypoints, scene_width, scene_height):
    b_accumulator = np.zeros((scene_height, scene_width), dtype=int)

    for match in m:
        box_kp = box_keypoints[match.queryIdx]
        scene_kp = scene_keypoints[match.trainIdx]

        vector = np.subtract(b, box_kp.pt)
        vector_scaled = np.multiply(vector, (scene_kp.size / box_kp.size))

        vector_reshaped = np.reshape(vector_scaled, (2,1))
        rotation_to_apply = scene_kp.angle - box_kp.angle
        vector_scaled_rotated = rotate(vector_reshaped, math.radians(rotation_to_apply))
        vector_scaled_rotated = vector_scaled_rotated.reshape(2)

        calc_bar = [int(round(scene_kp.pt[0]+vector_scaled_rotated[0])),
                    int(round(scene_kp.pt[1]+vector_scaled_rotated[1]))]

        if 0 <= calc_bar[0] < b_accumulator.shape[1] \
                and 0 <= calc_bar[1] < b_accumulator.shape[0]:
            b_accumulator[calc_bar[1], calc_bar[0]] += 1

    return b_accumulator


# returns the N max indexes couples in a
def n_max(a, n):
    width = a.shape[1]
    result = []
    arr = a.reshape((a.shape[0] * a.shape[1]))
    max_indexes = arr.argsort()[-n:][::-1]
    for index in max_indexes:
        row = index // width
        col = index % width
        if a[row, col] > 0:
            result.append((row, col))

    return result


def compute_barycenter(keypoints):
    x = np.mean([i.pt[0] for i in keypoints])
    y = np.mean([i.pt[1] for i in keypoints])
    return [x, y]


def remove_noise(barycenter_accumulator):
    visualization.display_img((barycenter_accumulator / np.max(barycenter_accumulator) * 255).astype(np.uint8))
    # Find coordinates of points that received at least one vote
    points = cv2.findNonZero(barycenter_accumulator)
    points = points.reshape((points.shape[0], 2))
    new_accumulator = np.zeros(barycenter_accumulator.shape, dtype=np.uint8)

    dist = 10

    for x, y in points:
        votes_count = np.sum(barycenter_accumulator[y-dist:y+dist, x-dist:x+dist])
        # If there aren't at least MIN_VOTES votes in the surrounding patch and in the same position, remove the vote
        if votes_count - 1 >= 2:
            new_accumulator[y, x] = 255

    return new_accumulator


# Performs a dilation to group near pixels in a blob, of which we compute the barycenter
def find_centers(barycenter_accumulator):
    visualization.display_img(barycenter_accumulator)
    centers = []

    dist = 40
    # Dilation
    maxima = n_max(barycenter_accumulator, 1000)

    for y, x in maxima:
        curr_vote = barycenter_accumulator[y, x]
        if curr_vote != 0:
            barycenter_accumulator[y - dist:y + dist, x - dist:x + dist] = 0
            barycenter_accumulator[y, x] = curr_vote
            centers.append((x, y))

    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31,31))
    #kernel = np.ones((3, 3), np.uint8)
    #barycenter_accumulator = cv2.dilate(barycenter_accumulator, kernel=kernel, iterations=1)
    visualization.display_img(barycenter_accumulator)

    # Find contours
    #contours, hierarchy = cv2.findContours(barycenter_accumulator, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Compute centers of the contours
    #centers = []
    #for cnt in contours:
    #    M = cv2.moments(cnt)
    #    cx = int(M['m10'] / M['m00'])
    #    cy = int(M['m01'] / M['m00'])
    #    centers.append((cx, cy))

    # Search point with more votes surrounding the centers

    return centers


def prove():
    template_path = '../images/object_detection_project/models/0.jpg'
    scene_path = '../images/object_detection_project/scenes/h2.jpg'

    template_color = load_images.load_img_color(template_path)
    template = image_processing.convert_grayscale(template_color)
    template = image_processing.equalize_histogram(template)
    if template.shape[0] >= 400:
        template = image_processing.blur_image(template)

    scene_color = load_images.load_img_color(scene_path)
    scene = image_processing.convert_grayscale(scene_color)
    scene = image_processing.equalize_histogram(scene)
    scene = image_processing.sharpen_img(scene)

    kp_t, des_t = feature_detection.detect_features_SIFT(template)

    kp_s, des_s = feature_detection.detect_features_SIFT(scene)
    matches = feature_matching.find_matches(des_t, des_s)
    barycenter = compute_barycenter(kp_t)
    barycenter_accumulator = compute_joining_vectors_table(matches, barycenter, kp_t, kp_s, scene.shape[1],
                                                           scene.shape[0])

    barycenter_accumulator = remove_noise(barycenter_accumulator)
    centers = find_centers(barycenter_accumulator)

    result = np.zeros(scene_color.shape, dtype=np.uint8)
    result = cv2.addWeighted(scene_color, 0.5, result, 0.5, 0)
    for c in centers:
        cv2.circle(result, c, 20, (0,0,255), 3)
        cv2.circle(result, c, 3, (0,255,0), -1)

    #visualization.display_img(image_processing.resize_img(barycenter_accumulator.astype(np.uint8), 2))
    visualization.display_img(image_processing.resize_img(result, 2))

prove()