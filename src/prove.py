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
    img = np.zeros((scene.shape[0], scene.shape[1], 3), dtype=np.uint8)

    joining_vectors_d = {}
    for match in m:
        box_kp = box_keypoints[match.queryIdx]
        scene_kp = scene_keypoints[match.trainIdx]

        vector = np.subtract(b, box_kp.pt)
        vector_scaled = np.multiply(vector, (scene_kp.size / box_kp.size))

        vector_reshaped = np.reshape(vector_scaled, (2,1))
        rotation_to_apply = scene_kp.angle - box_kp.angle
        vector_scaled_rotated = rotate(vector_reshaped, math.radians(rotation_to_apply))

        joining_vectors_d[match.trainIdx] = (match.queryIdx, np.rint(vector_scaled_rotated))

        #p1 = (int(scene_kp.pt[0]), int(scene_kp.pt[1]))
        #p2 = (int(p1[0] + vector_scaled_rotated[0]), int(p1[1] + vector_scaled_rotated[1]))
        #cv2.line(img, p1, p2, (50, 50, 50), 1)
        #cv2.circle(img, p1, 1, (0, 255, 0), -1)
        #cv2.rectangle(img, p2, (p2[0]+2, p2[1]+2), (0, 0, 255), -1)

    #visualization.display_img(image_processing.resize_img(img, 2))
    return joining_vectors_d


def compute_barycenter(keypoints):
    x = np.mean([i.pt[0] for i in keypoints])
    y = np.mean([i.pt[1] for i in keypoints])
    # Rimuovere astype(int)
    return np.rint([x, y]).astype(int)


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
    p2 = (int(p1[0] + v[0][0]), int(p1[1] + v[1][0]))
    cv2.line(img3, p1, p2, (50, 50, 50), 1)

for train_idx in joining_vectors_dict:
    query_index, v = joining_vectors_dict[train_idx]
    point = kp_s[train_idx].pt
    p1 = (int(point[0]), int(point[1]))
    p2 = (int(p1[0] + v[0][0]), int(p1[1] + v[1][0]))
    cv2.circle(img3, p1, 1, (0, 255, 0), -1)
    cv2.rectangle(img3, p2, (p2[0]+2, p2[1]+2), (0, 0, 255), -1)

visualization.display_img(image_processing.resize_img(img3, 2))
