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
    joining_vectors_d = {}
    for match in m:
        box_kp = box_keypoints[match.queryIdx]
        scene_kp = scene_keypoints[match.trainIdx]

        vector = np.subtract(b, box_kp.pt)
        vector_scaled = np.multiply(vector, (scene_kp.size / box_kp.size))

        vector_reshaped = np.reshape(vector_scaled, (2,1))
        rotation_to_apply = scene_kp.angle - box_kp.angle
        vector_scaled_rotated = rotate(vector_reshaped, math.radians(rotation_to_apply))
        vector_scaled_rotated = vector_scaled_rotated.reshape(2)
        joining_vectors_d[match.trainIdx] = (match.queryIdx, vector_scaled_rotated)

    return joining_vectors_d


def compute_accumulator(joining_vectors, sceneImg_shape, kp_scene):
    accumulator = np.zeros((sceneImg_shape[0], sceneImg_shape[1]))
    for i in joining_vectors:
        accum_i = joining_vectors[i][1][0] + kp_scene[i].pt[0]
        accum_j = joining_vectors[i][1][1] + kp_scene[i].pt[1]

        if 0 <= accum_i < accumulator.shape[1] and 0 <= accum_j < accumulator.shape[0]:
            accumulator[int(round(accum_j)), int(round(accum_i))] += 1

    return accumulator


# returns the N max elements and indices in a
def n_max(a, n):
    indices = a.ravel().argsort()[-n:]
    indices = (np.unravel_index(i, a.shape) for i in indices)
    return [(a[i], i) for i in indices]


def compute_barycenter(keypoints):
    x = np.mean([i.pt[0] for i in keypoints])
    y = np.mean([i.pt[1] for i in keypoints])
    return [x, y]


template_path = '../images/hp.jpg'
scene_path = '../images/hp_2.jpg'
template = load_images.load_img_color(template_path)
scene = load_images.load_img_color(scene_path)

sift = xfeatures2d.SIFT_create()
kp_t, des_t = feature_detection.detect_features_SIFT(template)
kp_s, des_s = feature_detection.detect_features_SIFT(scene)

matches = feature_matching.find_matches(des_t, des_s)

barycenter = compute_barycenter(kp_t)
barycenter_scene = compute_barycenter(kp_s)

joining_vectors_dict = compute_joining_vectors_table(matches, barycenter, kp_t, kp_s)

img3 = np.zeros(scene.shape, dtype=np.uint8)
for train_idx in joining_vectors_dict:
    query_index, v = joining_vectors_dict[train_idx]
    point = kp_s[train_idx].pt
    p1 = (int(point[0]), int(point[1]))
    p2 = (int(p1[0] + v[0]), int(p1[1] + v[1]))
    cv2.line(img3, p1, p2, (50, 50, 50), 1)

for train_idx in joining_vectors_dict:
    query_index, v = joining_vectors_dict[train_idx]
    point = kp_s[train_idx].pt
    p1 = (int(point[0]), int(point[1]))
    p2 = (int(p1[0] + v[0]), int(p1[1] + v[1]))
    cv2.circle(img3, p1, 1, (0, 255, 0), -1)
    cv2.rectangle(img3, p2, (p2[0]+2, p2[1]+2), (0, 0, 255), -1)

visualization.display_img(image_processing.resize_img(img3, 2))


accumulator = compute_accumulator(joining_vectors_dict, scene.shape, kp_s)
print(accumulator)

visualization.display_img(accumulator)
