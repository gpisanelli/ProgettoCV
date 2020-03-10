import cv2
import numpy as np

import image_processing
import load_images
import visualization


def compute_barycenter(keypoints):
    x = np.mean([i.pt[0] for i in keypoints])
    y = np.mean([i.pt[1] for i in keypoints])
    return (x, y)


def compute_joining_vectors_table(matches, barycenter, keypoints):
    joining_vectors_dict = {}
    for match in matches:
        # table element[sceneIndex] = templateIndex, vector_tuple
        joining_vectors_dict[match.trainIdx] = [match.queryIdx,
                                                (barycenter[0] - keypoints[match.queryIdx].pt[0],
                                                 barycenter[1] - keypoints[match.queryIdx].pt[1])]
    return joining_vectors_dict


def compute_accumulator(joining_vectors, sceneImg_shape, kp_scene):
    accumulator = np.zeroes(sceneImg_shape)
    for i in range(len(kp_scene)):
        accum_i, accum_j = joining_vectors[i][1][0] + kp_scene[i].pt[0], joining_vectors[i][1][1] + kp_scene[i].pt[1]
        if accum_i < accumulator.shape[0] and accum_j < accumulator.shape[1]:
            accumulator[accum_i, accum_j] += 1


    return accumulator


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