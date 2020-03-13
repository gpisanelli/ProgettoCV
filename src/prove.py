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
import sklearn as sk
from sklearn.cluster import KMeans


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


def opencv_kmeans(points):
    from matplotlib import pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')

    Z = np.float32(points)
    # Z = np.float32(n_max(barycenter_accumulator, 50))
    # print(Z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    max_range = 11 if len(points) > 10 else len(points) + 1
    compactnesses = {}
    labels = {}
    centers = {}
    for k in range(1, max_range):
        print('k = ', k)
        compactness, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        compactnesses[k] = compactness
        labels[k] = label
        centers[k] = center

    max_key_compactnesses = max(compactnesses, key=compactnesses.get)
    best_compactness = compactnesses[max_key_compactnesses]
    best_label = labels[max_key_compactnesses]
    best_center = centers[max_key_compactnesses]

    print('Best compactness = ', best_compactness)
    print('Best label = ', best_label)
    print('Best center = ', best_center)

    # Now separate the data, Note the flatten()
    A = Z[best_label.ravel() == 0]
    B = Z[best_label.ravel() == 1]

    # Plot the data
    plt.scatter(A[:, 0], A[:, 1])
    plt.scatter(B[:, 0], B[:, 1], c='r')
    plt.scatter(best_center[:, 0], best_center[:, 1], s=80, c='y', marker='s')
    plt.xlabel('x'), plt.ylabel('y')
    plt.show()

    return best_center


def sklearn_kmeans(points):
    X = np.array(points)
    # print(X)
    # kmeans in sklearn doesn't allow a cluster per point, that's why range doesn't have +1 in the else
    max_range = 11 if len(points) > 10 else len(points)
    models = {}
    silhouettes = {}
    calinskis = {}
    davieses = {}
    for k in range(1, max_range):
        kmeans = KMeans(n_clusters=k).fit(X)
        labels = kmeans.labels_
        models[k] = kmeans
        # metrics can't score a single cluster :/
        if len(np.unique(labels)) == 1:
            # silhouette varies between -1 and 1, 0.9 is pretty high
            silhouettes[k] = 0.9
            # calinski is better if higher (?)
            calinskis[k] = 300
            # davis is better if lower (?)
            davieses[k] = 0.1
        else:
            silhouettes[k] = sk.metrics.silhouette_score(X, labels, metric='euclidean')
            calinskis[k] = sk.metrics.calinski_harabasz_score(X, labels)
            davieses[k] = sk.metrics.davies_bouldin_score(X, labels)

    # higher score is better
    max_key_silhouettes = max(silhouettes, key=silhouettes.get)
    best_model_silhouette = models[max_key_silhouettes]
    print('Best silhouette = ', silhouettes[max_key_silhouettes])
    print('Best silhouette labels = ', best_model_silhouette.labels_)
    print('Bets silhouette centroids = ', best_model_silhouette.cluster_centers_)
    # higher score is better
    max_key_calinskis = max(calinskis, key=calinskis.get)
    best_model_calinski = models[max_key_calinskis]
    print('Best calinski = ', calinskis[max_key_calinskis])
    print('Best calinski labels = ', best_model_calinski.labels_)
    print('Bets calinski centroids = ', best_model_calinski.cluster_centers_)
    # lower score is better
    min_key_davieses = min(davieses, key=davieses.get)
    best_model_davies = models[min_key_davieses]
    print('Best davis = ', davieses[min_key_davieses])
    print('Best davis labels = ', best_model_davies.labels_)
    print('Bets davis centroids = ', best_model_davies.cluster_centers_)

    return best_model_silhouette.cluster_centers_, best_model_calinski.cluster_centers_, best_model_davies.cluster_centers_



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

maxima = n_max(barycenter_accumulator, 100)
#result = cv2.addWeighted(cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY), 1, (np.divide(barycenter_accumulator, np.max(barycenter_accumulator)) * 255).astype(np.uint8), 1, 0)
#visualization.display_img(image_processing.resize_img(result, 2))
#whites = np.copy(barycenter_accumulator)
#whites[whites > 0] = 255
#visualization.display_img(whites.astype(np.uint8))

#visualization.display_img(thr_acc.astype(np.uint8))


thr_acc = (barycenter_accumulator > 2) * barycenter_accumulator
points_reached_by_at_least_two_vectors = np.argwhere(thr_acc > 0)

silhouette_centroids, calinski_centroids, davies_centroids = sklearn_kmeans(points_reached_by_at_least_two_vectors)
opencv_centroids = opencv_kmeans(points_reached_by_at_least_two_vectors)
