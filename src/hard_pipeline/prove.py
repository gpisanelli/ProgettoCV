import math
import time

import cv2
import numpy as np

import feature_detection
import feature_matching
import image_processing
import load_images
import object_validation
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


def filter_matches(matches_barycenters, barycenters):
    good_matches = {}
    max_dist = 80

    for i in range(len(barycenters)):
        good_matches[i] = (barycenters[i], [])

    for match, vector in matches_barycenters:
        found = False
        i = 0
        while not found and i < len(barycenters):
            b_x, b_y = barycenters[i]
            if math.sqrt((vector[0] - b_x) ** 2 + (vector[1] - b_y) ** 2) <= max_dist:
                good_matches[i][1].append(match)
                found = True
            i += 1

    return good_matches


# returns barycenter accumulator and a list of (matches, supposed_barycenter) useful for filter_matches function
def compute_barycenter_accumulator(m, b, box_keypoints, scene_keypoints, scene_width, scene_height):
    b_accumulator = np.zeros((scene_height, scene_width), dtype=int)
    match_barycenter_list = []

    for match in m:
        box_kp = box_keypoints[match.queryIdx]
        scene_kp = scene_keypoints[match.trainIdx]

        vector = np.subtract(b, box_kp.pt)
        vector_scaled = np.multiply(vector, (scene_kp.size / box_kp.size))

        vector_reshaped = np.reshape(vector_scaled, (2, 1))
        rotation_to_apply = scene_kp.angle - box_kp.angle
        vector_scaled_rotated = rotate(vector_reshaped, math.radians(rotation_to_apply))
        vector_scaled_rotated = vector_scaled_rotated.reshape(2)

        bar_x = int(round(scene_kp.pt[0] + vector_scaled_rotated[0]))
        bar_y = int(round(scene_kp.pt[1] + vector_scaled_rotated[1]))
        calc_bar = [bar_x, bar_y]

        match_barycenter_list.append((match, (bar_x, bar_y)))

        if 0 <= calc_bar[0] < b_accumulator.shape[1] \
                and 0 <= calc_bar[1] < b_accumulator.shape[0]:
            b_accumulator[calc_bar[1], calc_bar[0]] += 1

    return b_accumulator, match_barycenter_list


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
            result.append(((row, col), a[row, col]))

    return result


def compute_barycenter(keypoints):
    x = np.mean([i.pt[0] for i in keypoints])
    y = np.mean([i.pt[1] for i in keypoints])
    return [x, y]


def remove_noise(barycenter_accumulator):
    #visualization.display_img((barycenter_accumulator / np.max(barycenter_accumulator) * 255).astype(np.uint8))
    # Find coordinates of points that received at least one vote
    points = cv2.findNonZero(barycenter_accumulator)
    points = points.reshape((points.shape[0], 2))
    new_accumulator = np.zeros(barycenter_accumulator.shape, dtype=np.uint8)

    dist = 20

    for x, y in points:
        votes_count = np.sum(barycenter_accumulator[y - dist:y + dist, x - dist:x + dist])
        # If there aren't at least MIN_VOTES votes in the surrounding patch and in the same position, remove the vote
        if votes_count - 1 >= 2:
            new_accumulator[y, x] = barycenter_accumulator[y, x]

    return new_accumulator


def find_centers1(barycenter_accumulator):
    centers = []
    #visualization.display_img((barycenter_accumulator / np.max(barycenter_accumulator) * 255).astype(np.uint8),
    #                          title='Before')
    dist = 50
    maxima = n_max(barycenter_accumulator, 1000)
    maxima[::-1].sort(key=lambda m: m[1])

    for m in maxima:
        y, x = m[0]
        votes = barycenter_accumulator[y, x]

        if votes > 0:
            for row in range(y - dist, y + dist):
                for col in range(x - dist, x + dist):
                    if 0 <= row < barycenter_accumulator.shape[0] and 0 <= col < barycenter_accumulator.shape[1]:
                        barycenter_accumulator[row, col] = 0
            barycenter_accumulator[y, x] = votes
            centers.append((x, y))

    return centers


# Performs a dilation to group near pixels in a blob, of which we compute the barycenter
def find_centers(barycenter_accumulator):
    #visualization.display_img((barycenter_accumulator / np.max(barycenter_accumulator) * 255).astype(np.uint8))
    barycenter_accumulator[barycenter_accumulator > 0] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    barycenter_accumulator = cv2.dilate(barycenter_accumulator, kernel=kernel, iterations=1)
    #visualization.display_img((barycenter_accumulator / np.max(barycenter_accumulator) * 255).astype(np.uint8))

    # Find contours
    contours, hierarchy = cv2.findContours(barycenter_accumulator, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Compute centers of the contours
    centers = []
    for cnt in contours:
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        centers.append((cx, cy))

    # Search point with more votes surrounding the centers

    return centers


def compute_hough(template, kp_t, des_t, scene, kp_s, des_s, color_template, color_scene):
    matches = feature_matching.find_matches(des_t, des_s)
    result_bounds = []

    if len(matches) >= 10:
        #matches_scene = cv2.drawMatches(template, kp_t, scene, kp_s, matches, None, (0,0,255))
        #[visualization.display_img(matches_scene)

        barycenter = compute_barycenter(kp_t)
        barycenter_accumulator, matches_barycenters = \
            compute_barycenter_accumulator(matches, barycenter, kp_t, kp_s, scene.shape[1], scene.shape[0])

        if cv2.countNonZero(barycenter_accumulator) > 0:
            barycenter_accumulator = remove_noise(barycenter_accumulator)
            centers = find_centers(barycenter_accumulator)

            good_matches = filter_matches(matches_barycenters, centers)

            for i in good_matches:
                if len(good_matches[i][1]) >= 4:
                    success, bounds, M, used_src_pts, used_dst_pts, not_used_matches = feature_matching.find_object(good_matches[i][1],
                                                                                                           kp_t, kp_s, template)
                    if success:
                        # Object validation
                        polygon_convex = object_validation.is_convex_polygon(bounds)
                        if polygon_convex:
                            color_validation = object_validation.validate_color(color_template, color_scene, used_src_pts,
                                                                                    used_dst_pts, bounds, M)
                            if color_validation:
                                x, y, w, h = cv2.boundingRect(bounds)
                                result_bounds.append(bounds)
                    #good_matches[i] = (good_matches[i][0], not_used_matches)

    return result_bounds
