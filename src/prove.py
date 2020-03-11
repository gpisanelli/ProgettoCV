import math

import cv2
import numpy as np

import feature_detection
import feature_matching
import image_processing
import load_images
import visualization
import cv2
from cv2 import xfeatures2d


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



def compute_joining_vectors_table(m, b, box_keypoints, scene_keypoints):
    barycenter_accumulator = np.zeros((scene.shape[0], scene.shape[1]), dtype=int)

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

        if 0 <= calc_bar[0] < barycenter_accumulator.shape[1] and 0 <= calc_bar[1] < barycenter_accumulator.shape[0]:
            barycenter_accumulator[calc_bar[1], calc_bar[0]] += 1

    return barycenter_accumulator


def compute_accumulator(joining_vectors, sceneImg_shape, kp_scene):
    accumulator = np.zeros(sceneImg_shape)
    for i in joining_vectors:
        accum_i = joining_vectors[i][1][0] + kp_scene[i].pt[0]
        accum_j = joining_vectors[i][1][1] + kp_scene[i].pt[1]

        if 0 <= accum_i < accumulator.shape[1] and 0 <= accum_j < accumulator.shape[0]:
            accumulator[int(round(accum_j)), int(round(accum_i))] += 1

    return accumulator


# returns the N max indexes couples in a
def n_max(a, n):
    width = a.shape[1]
    result = []
    arr = a.reshape((a.shape[0] * a.shape[1]))
    max_indexes = arr.argsort()[-n:][::-1]
    for index in max_indexes:
        row = index // width
        col = index % width
        result.append((row, col))
    return result


def compute_barycenter(keypoints):
    x = np.mean([i.pt[0] for i in keypoints])
    y = np.mean([i.pt[1] for i in keypoints])
    return [x, y]


template_path = '../images/object_detection_project/models/0.jpg'
scene_path = '../images/object_detection_project/scenes/e1.png'
template = load_images.load_img_color(template_path)
scene = load_images.load_img_color(scene_path)

sift = xfeatures2d.SIFT_create()
kp_t, des_t = feature_detection.detect_features_SIFT(template)
kp_s, des_s = feature_detection.detect_features_SIFT(scene)

matches = feature_matching.find_matches(des_t, des_s)

barycenter = compute_barycenter(kp_t)
barycenter_scene = compute_barycenter(kp_s)

barycenter_accumulator = compute_joining_vectors_table(matches, barycenter, kp_t, kp_s)
#visualization.display_img(image_processing
#                          .resize_img((np.divide(barycenter_accumulator, np.max(barycenter_accumulator)) * 255)
#                                      .astype(np.uint8), 2))

#accumulator = compute_accumulator(joining_vectors_dict, scene.shape, kp_s)

maxima = n_max(barycenter_accumulator, 5)

result = cv2.addWeighted(cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY), 1, (np.divide(barycenter_accumulator, np.max(barycenter_accumulator)) * 255).astype(np.uint8), 1, 0)
visualization.display_img(image_processing.resize_img(result, 2))
