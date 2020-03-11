import cv2
import numpy as np

import image_processing
import load_images
import visualization


# matches lista dMatch -> ogni dMatch è un oggetto con 4 elementi ->
# -> distance tra i keypoint
# -> imageIndex (sempre 0)
# -> queryIndex (indice del descriptor nella lista del template)
# -> trainIndex (indice del descriptor nella lista della scena)
# Dato che passiamo alla ricerca dei matches le liste dei descrittori
# ottenute da sift e che queste hanno i descrittori nello stesso ordine in
# cui sono i relativi keypoint (vedi sift.detectAndCompute()) ->
# -> se manteniamo nella struttura dei joining vectors le corrispondenze
# tra indici possiamo risalire accedendo con essi alle liste ottenute da
# sift per ricavare i matching keypoint/descriptor -> possiamo quindi
# proseguire col processo di votazione sulla scena


# https://answers.opencv.org/question/18436/what-to-do-with-dmatch-value/
# http://www.programmersought.com/article/3827388529/


def compute_barycenter(keypoints):
    x = np.mean([i.pt[0] for i in keypoints])
    y = np.mean([i.pt[1] for i in keypoints])
    return [x, y]


def compute_joining_vectors_table(matches, barycenter, box_keypoints, scene_keypoints):
    joining_vectors_dict = {}
    for match in matches:
        box_kp = box_keypoints[match.queryIdx]
        # table element[sceneIndex] = templateIndex, vector_tuple
        vector = np.subtract(barycenter, box_kp.pt)
        vector_scaled = vector * (box_kp.size / scene_keypoints[match.trainIdx].size)

        vector_reshaped = np.reshape(vector_scaled, (2, 1))
        rotation_to_apply = box_kp.angle - scene_keypoints[match.trainIdx].angle
        vector_scaled_rotated = rotate(vector_reshaped, rotation_to_apply)

        joining_vectors_dict[match.trainIdx] = (match.queryIdx, np.rint(vector_scaled_rotated))

    return joining_vectors_dict


# (cos(θ) −sin(θ))    (x)
# (sin(θ) cos(θ) ) *  (y)
# theta: angle in radians
# vector: numpy array
def rotate(theta, vector):
    # Create rotation matrix
    rot_matrix = np.array(((np.cos(theta), -np.sin(theta)),
                           (np.sin(theta), np.cos(theta))))

    # Apply the rotation matrix to the vector
    return rot_matrix.dot(vector)


def compute_accumulator(joining_vectors, sceneImg_shape, kp_scene):
    accumulator = np.zeroes(sceneImg_shape)
    for i in range(len(kp_scene)):
        accum_i = joining_vectors[i][1][0] + kp_scene[i].pt[0]
        accum_j = joining_vectors[i][1][1] + kp_scene[i].pt[1]
        if accum_i < accumulator.shape[0] and accum_j < accumulator.shape[1]:
            accumulator[accum_i, accum_j] += 1

    return accumulator


# returns the N max elements and indices in a
def n_max(a, n):
    indices = a.ravel().argsort()[-n:]
    indices = (np.unravel_index(i, a.shape) for i in indices)
    return [(a[i], i) for i in indices]


def compute_GHT():
    pass

def main():
    hough = cv2.createGeneralizedHoughGuil()
    hough.setMaxAngle(5)
    hough.setMinAngle(0)
    hough.setAngleStep(1)
    hough.setMaxScale(4)
    hough.setMinScale(0.1)
    hough.setScaleStep(0.01)
    hough.setMinDist(10)
    hough.setLevels(100)
    hough.setAngleThresh(1000)
    hough.setScaleThresh(100)
    hough.setPosThresh(100)
    hough.setDp(2)
    hough.setMaxBufferSize(1000)

    template_path = '../images/generalized_hough_demo_02.png'
    template = load_images.load_img_grayscale(template_path)

    scene_path = '../images/generalized_hough_demo_01.png'
    scene = load_images.load_img_grayscale(scene_path)

    hough.setTemplate(template, (int(template.shape[1] / 2), int(template.shape[0] / 2)))
    positions, votes = hough.detect(scene)

    for position in positions:
        print(position)
        cv2.polylines()
        cv2.circle(scene, (int(position[0][0]), int(position[0][1])), 10, 100, 3)

    visualization.display_img(scene)


main()