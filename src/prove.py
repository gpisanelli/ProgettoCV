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
        # table element[sceneIndex] = templateIndex, vector_tuple
        vector = np.subtract(b, box_kp.pt)
        vector_scaled = vector * (box_kp.size / scene_keypoints[match.trainIdx].size)

        vector_reshaped = np.reshape(vector_scaled, (2,1))
        rotation_to_apply = box_kp.angle - scene_keypoints[match.trainIdx].angle
        vector_scaled_rotated = rotate(vector_reshaped, rotation_to_apply)

        joining_vectors_d[match.trainIdx] = (match.queryIdx, np.rint(vector_scaled_rotated))

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

joining_vectors_dict = compute_joining_vectors_table(matches[:10], barycenter, kp_t, kp_s)
print(len(kp_s))
img3 = np.zeros(scene.shape, dtype=np.uint8)
for train_idx in joining_vectors_dict:
    query_index, v = joining_vectors_dict[train_idx]
    point = kp_s[train_idx].pt
    p1 = (int(point[0]), int(point[1]))
    print(v[1][0])
    p2 = (int(p1[0] + v[0][0]), int(p1[1] + v[1][0]))
    cv2.line(img3, p1, (barycenter_scene[0]-p1[0],barycenter_scene[1]-p1[1]), (255,0,0), 1)

cv2.circle(img3, (barycenter_scene[0],barycenter_scene[1]), 5, (0,0,255), -1)
visualization.display_img(image_processing.resize_img(img3, 2))
