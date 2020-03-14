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


def prove():
    template_path = '../images/object_detection_project/models/13.jpg'
    scene_path = '../images/object_detection_project/scenes/h1.jpg'

    template_color = load_images.load_img_color(template_path)
    template = image_processing.convert_grayscale(template_color)
    template = image_processing.equalize_histogram(template)

    scene_color = load_images.load_img_color(scene_path)
    scene = image_processing.convert_grayscale(scene_color)
    scene = image_processing.equalize_histogram(scene)
    scene = image_processing.sharpen_img(scene)

    kp_t, des_t = feature_detection.detect_features_SIFT(template)

    kp_s, des_s = feature_detection.detect_features_SIFT(scene)
    matches = feature_matching.find_matches(des_t, des_s)
    barycenter = compute_barycenter(kp_t)
    barycenter_accumulator, matches_barycenters = \
        compute_barycenter_accumulator(matches, barycenter, kp_t, kp_s, scene.shape[1], scene.shape[0])

    barycenter_accumulator = remove_noise(barycenter_accumulator)
    centers = find_centers(barycenter_accumulator)

    good_matches = filter_matches(matches_barycenters, centers)

    result = np.zeros(scene_color.shape, dtype=np.uint8)
    result = cv2.addWeighted(scene_color, 0.5, result, 0.5, 0)

    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 0, 255),
        (0, 255, 255),
        (255, 255, 0),
        (255, 255, 255),
        (255, 255, 255),
        (255, 255, 255),
        (255, 255, 255),
        (255, 255, 255),
        (255, 255, 255),
        (255, 255, 255),
        (255, 255, 255)
    ]

    for i in good_matches:
        for m in good_matches[i][1]:
            pt = kp_s[m.trainIdx].pt
            cv2.circle(result, (int(pt[0]), int(pt[1])), 2, colors[i], -1)

    visualization.display_img(result)

    visualization_scene = scene_color.copy()
    for i in good_matches:
        print('Len good_matches[{}][1] = {}'.format(i, len(good_matches[i][1])))
        if len(good_matches[i][1]) >= 6:
            bounds, M, used_src_pts, used_dst_pts, not_used_matches = feature_matching.find_object(good_matches[i][1],
                                                                                                   kp_t, kp_s, template)

            # Object validation
            polygon_convex = object_validation.is_convex_polygon(bounds)
            if polygon_convex:
                visualization_scene = visualization.draw_polygons(visualization_scene, [bounds])
        elif len(good_matches[i][1]) > 0:
            # disegno rettangolo approssimativo perché not enough points per la homography
            # cerco la miglior scale e rotation facendo la media tra quelle disponibili
            scales = []
            rotations = []
            for j in range(len(good_matches[i][1])):
                scales.append(kp_s[good_matches[i][1][j].trainIdx].size / kp_t[good_matches[i][1][j].queryIdx].size)
                rotations.append(kp_s[good_matches[i][1][j].trainIdx].angle - kp_t[good_matches[i][1][j].queryIdx].angle)
            scale_factor = np.mean(scales)
            rotation_factor = np.mean(rotations)
            print('Scale mean {} = {}'.format(i, scale_factor))
            print('Rotation mean {} = {}'.format(i, rotation_factor))

            # computo i vettori congiungenti il centro del template ai suoi vertici (ordine importante per polylines)
            box_vertexes = [(0, 0), (template.shape[1], 0), (template.shape[1], template.shape[0]), (0, template.shape[0])]
            scene_vertexes = []
            vectors = []
            int_barycenter = np.int32(barycenter)
            for j in range(4):
                vectors.append(np.subtract(box_vertexes[j], int_barycenter))

            # print e display per controllare come mai non va icsdi
            print('Box vertexes {} = {}'.format(i, box_vertexes))
            print('Box vectors {} = {}'.format(i, vectors))
            template_copy = template.copy()
            for j in range(4):
                cv2.circle(template_copy, box_vertexes[j], 20, (0, 255, 0), -1)
                cv2.circle(template_copy, (int(barycenter[0]), int(barycenter[1])), 20, (0, 255, 0), -1)
                cv2.arrowedLine(template_copy,
                                (int(barycenter[0]), int(barycenter[1])),
                                (int(barycenter[0] + vectors[j][0]), int(barycenter[1] + vectors[j][1])), (0, 255, 0))

            visualization.display_img(template_copy, title='Template_vertexes_vectors')

            # scalo e ruoto i vettori secondo le scale e rotation prima trovate
            for j in range(4):
                vectors[j] = np.multiply(vectors[j], scale_factor)
                vectors[j] = np.reshape(vectors[j], (2, 1))
                vectors[j] = rotate(vectors[j], math.radians(rotation_factor))
                vectors[j] = vectors[j].reshape(2)

                cv2.arrowedLine(template_copy,
                                (int(barycenter[0]), int(barycenter[1])),
                                (int(barycenter[0] + vectors[j][0]), int(barycenter[1] + vectors[j][1])), (0, 255, 0))

                # computo i vertici risultanti nella scena
                scene_vertex_x = int(round(good_matches[i][0][0] + vectors[j][0]))
                scene_vertex_y = int(round(good_matches[i][0][1] + vectors[j][1]))
                scene_vertexes.append((scene_vertex_x, scene_vertex_y))

            # print e display per controllare come mai non va icsdi
            print('Scene vertexes match {} = {}'.format(i, scene_vertexes))
            scene_copy = scene.copy()
            for j in range(4):
                cv2.circle(scene_copy, (int(good_matches[i][0][0]), int(good_matches[i][0][1])), 10, (0, 255, 0), -1)
                cv2.circle(scene_copy, scene_vertexes[j], 10, (0, 255, 0), -1)
                cv2.arrowedLine(scene_copy, scene_vertexes[j], (int(round(scene_vertexes[j][0]-vectors[j][0])),
                                                                int(round(scene_vertexes[j][1]-vectors[j][1]))), (0, 255, 0))
            visualization.display_img(scene_copy, title='Scene_vertexes_vectors')

            visualization_scene = visualization.draw_polygons(visualization_scene, np.int32([scene_vertexes]))

    visualization.display_img(visualization_scene)


prove()

'''
Qua sotto roba clustering prolly useless

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
        # metrics can't score a single cluster, i have to give an estimated score :/
        if len(np.unique(labels)) == 1:
            # silhouette varies between -1 and 1, 0.9 is pretty high
            silhouettes[k] = 0.9
            # calinski is better if higher (?)
            calinskis[k] = 300
            # davies is better if lower (?)
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


thr_acc = (barycenter_accumulator > 2) * barycenter_accumulator
visualization.display_img(thr_acc.astype(np.uint8))
points_reached_by_at_least_two_vectors = np.argwhere(thr_acc > 0)

# to be set to the wanted points
considered_points = maxima

length_considered_points = len(considered_points)
print('Length considered points = ', length_considered_points)
if length_considered_points > 1:
    # silhouette_centroids, calinski_centroids, davies_centroids = sklearn_kmeans(points_reached_by_at_least_two_vectors)
    opencv_centroids = opencv_kmeans(considered_points)
elif length_considered_points == 1:
    print('Single point present, centroid = ', considered_points[0])
'''
