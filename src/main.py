import time
import sys
import numpy as np
import cv2

import feature_detection
import feature_matching
import hough_detection
import image_processing
import load_images
import object_validation
import visualization
import parallel_hough

# Scenes dictionary
scene_dict = {
    '-e': ['e1.png', 'e2.png', 'e3.png', 'e4.png', 'e5.png'],
    '-m': ['m1.png', 'm2.png', 'm3.png', 'm4.png', 'm5.png'],
    '-h': ['h1.jpg', 'h2.jpg', 'h3.jpg', 'h4.jpg', 'h5.jpg']
}

# Image loading
# scene_names = [
#    'e1.png','e2.png','e3.png','e4.png','e5.png'
#    #'m1.png','m2.png','m3.png','m4.png','m5.png'
#    #'h1.jpg','h2.jpg','h3.jpg','h4.jpg','h5.jpg'
# ]

box_dict = {
    '-e': ['0.jpg', '1.jpg', '11.jpg', '19.jpg', '24.jpg', '25.jpg', '26.jpg'],
    '-m': ['0.jpg', '1.jpg', '11.jpg', '19.jpg', '24.jpg', '25.jpg', '26.jpg'],
    '-h': ['0.jpg', '1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6.jpg',
           '7.jpg', '8.jpg', '9.jpg', '10.jpg', '11.jpg', '12.jpg', '13.jpg',
           '14.jpg', '15.jpg', '16.jpg', '17.jpg', '18.jpg', '19.jpg', '20.jpg',
           '21.jpg', '22.jpg', '23.jpg', '24.jpg', '25.jpg', '26.jpg']
}
box_names = ['0.jpg', '1.jpg', '11.jpg', '19.jpg', '24.jpg', '25.jpg', '26.jpg']

dict_box_features = {}


# Since in a realistic scenario we should already already have the images of the boxes, we can precompute their features
# and save them in a dictionary (or better initialize the dictionary from a file, which will be loaded on application
# startup). The features will later be retrieved from the dictionary, avoiding the cost of computing them during
# execution.
def precompute_box_features_easy():
    for box_name in box_dict['-e']:
        box_path = load_images.get_path_for_box(box_name)
        box = load_images.load_img_color(box_path)
        proc_box = preprocess_box_easy(box)
        kp_box, des_box = feature_detection.detect_features_SIFT(proc_box)
        dict_box_features[box_name] = (box, proc_box, kp_box, des_box)


def preprocess_box_easy(b):
    pr_box = b.copy()
    pr_box = image_processing.convert_grayscale(pr_box)
    pr_box = image_processing.equalize_histogram(pr_box)
    if pr_box.shape[1] >= 200:
        pr_box = image_processing.blur_image(pr_box)

    return pr_box


def preprocess_scene_easy(s):
    # Preprocessing (scene image)
    pr_scene = s.copy()
    pr_scene = image_processing.convert_grayscale(pr_scene)
    pr_scene = image_processing.equalize_histogram(pr_scene)
    pr_scene = image_processing.sharpen_img(pr_scene)

    return pr_scene


def precompute_box_features_medium():
    for box_name in box_dict['-m']:
        box_path = load_images.get_path_for_box(box_name)
        box = load_images.load_img_color(box_path)
        proc_box = preprocess_box_medium(box)
        kp_box, des_box = feature_detection.detect_features_SIFT(proc_box)
        dict_box_features[box_name] = (box, proc_box, kp_box, des_box)


def preprocess_box_medium(b):
    pr_box = b.copy()
    pr_box = image_processing.convert_grayscale(pr_box)
    pr_box = image_processing.equalize_histogram(pr_box)
    if pr_box.shape[1] >= 200:
        pr_box = image_processing.blur_image(pr_box)

    return pr_box


def preprocess_scene_medium(s):
    # Preprocessing (scene image)
    pr_scene = s.copy()
    pr_scene = image_processing.convert_grayscale(pr_scene)
    pr_scene = image_processing.equalize_histogram(pr_scene)
    pr_scene = image_processing.sharpen_img(pr_scene)

    return pr_scene


def precompute_box_features_hard():
    for box_name in box_dict['-h']:
        box_path = load_images.get_path_for_box(box_name)
        box = load_images.load_img_color(box_path)
        proc_box = preprocess_box_hard(box)
        kp_box, des_box = feature_detection.detect_features_SIFT(proc_box)
        dict_box_features[box_name] = (box, proc_box, kp_box, des_box)


def preprocess_box_hard(b):
    pr_box = b.copy()
    pr_box = image_processing.convert_grayscale(pr_box)
    pr_box = image_processing.equalize_histogram(pr_box)
    if pr_box.shape[1] >= 200:
        pr_box = image_processing.blur_image(pr_box)

    return pr_box


def main():
    accepted_list = ['-e', '-m', '-h']
    arg = sys.argv[1]
    if len(sys.argv) != 2 or arg not in accepted_list:
        print('usage: ./main.py -(e|h|m)')
        sys.exit(2)

    scene_names = scene_dict[arg]
    scenes = []

    for scene_name in scene_names:
        scene_path = load_images.get_path_for_scene(scene_name)
        scenes.append(load_images.load_img_color(scene_path))

    if arg == '-e':
        precompute_box_features_easy()
        for scene in scenes:
            proc_scene = preprocess_scene_easy(scene)
            test_scene = scene.copy()
            test_scene = image_processing.resize_img_dim(test_scene, proc_scene.shape[1], proc_scene.shape[0])
            visualization_scene = test_scene.copy()

            # s = time.time()
            kp_scene, des_scene = feature_detection.detect_features_SIFT(proc_scene)
            # print('\nTIME DETECTION: ', time.time() - s, '\n')

            for box_name in box_dict['-e']:
                # Box features retrieval
                box, proc_box, kp_box, des_box = dict_box_features[box_name]
                test_box = box.copy()
                test_box = image_processing.resize_img_dim(test_box, proc_box.shape[1], proc_box.shape[0])

                # Feature matching
                matches = feature_matching.find_matches(des_box, des_scene)

                searching = True
                while len(matches) > 10 and searching:
                    # Object detection
                    success, bounds, homography, used_box_points, used_scene_points, not_used_matches = \
                        feature_matching.find_object(matches, kp_box, kp_scene, proc_box)

                    if success:
                        # Object validation
                        polygon_convex = object_validation.is_convex_polygon(bounds)

                        if polygon_convex:
                            color_validation = object_validation.validate_color(test_box, test_scene, used_box_points,
                                                                                used_scene_points, bounds, homography)
                            if color_validation:
                                x, y, w, h = cv2.boundingRect(bounds)

                                if cv2.contourArea(bounds) / (w*h) >= 0.9:
                                    searching = False
                                    visualization_scene = visualization.draw_polygons(visualization_scene, [bounds])
                                    visualization_scene = visualization.draw_names(visualization_scene, bounds,
                                                                                   box_name)

                            matches = not_used_matches
                        else:
                            searching = False
                    else:
                        searching = False

            visualization.display_img(visualization_scene, 800, 'Result (press Esc to continue)')

    elif arg == '-m':
        precompute_box_features_medium()
        for scene in scenes:
            # forse preprocessing to change per le medium
            # template_color = load_images.load_img_color(template_path)
            # template = image_processing.convert_grayscale(template_color)
            # template = image_processing.equalize_histogram(template)

            scene_color = scene.copy()
            proc_scene = preprocess_scene_medium(scene_color)
            test_scene = scene_color.copy()
            test_scene = image_processing.resize_img_dim(test_scene, scene.shape[1], scene.shape[0])
            visualization_scene = test_scene.copy()  # necessarie tutte ste scene?

            # s = time.time()
            kp_s, des_s = feature_detection.detect_features_SIFT(proc_scene)
            # print('\nTIME DETECTION: ', time.time() - s, '\n')

            matches_mask = np.zeros(proc_scene.shape, dtype=np.uint8)

            color = 1
            bounds_dict = {}

            for box_name in box_dict['-m']:
                print('\nBOX: ', box_name)
                # Box features retrieval
                box, proc_box, kp_t, des_t = dict_box_features[
                    box_name]  # preprocessing dei box è uguale per tutte le scene?

                test_box = box.copy()
                test_box = image_processing.resize_img_dim(test_box, proc_box.shape[1],
                                                           proc_box.shape[0])  # necessario?

                matches = feature_matching.find_matches(des_t, des_s)
                barycenter = hough_detection.compute_barycenter(kp_t)
                barycenter_accumulator, matches_barycenters = hough_detection.compute_barycenter_accumulator(matches,
                                                                                                             barycenter,
                                                                                                             kp_t,
                                                                                                             kp_s,
                                                                                                             scene.shape[
                                                                                                                 1],
                                                                                                             scene.shape[
                                                                                                                 0])
                if cv2.countNonZero(barycenter_accumulator) > 0:
                    barycenter_accumulator = hough_detection.remove_noise(barycenter_accumulator)
                    centers = hough_detection.find_centers(barycenter_accumulator)

                    good_matches = hough_detection.filter_matches(matches_barycenters, centers)

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

                    for i in good_matches:
                        print('Len good_matches[{}][1] = {}'.format(i, len(good_matches[i][1])))
                        if len(good_matches[i][1]) >= 10:
                            success, bounds, homography, used_src_pts, used_dst_pts, not_used_matches = feature_matching.find_object(
                                good_matches[i][1],
                                kp_t, kp_s, proc_box)

                            if success:
                                # Object validation
                                polygon_convex = object_validation.is_convex_polygon(bounds)
                                if polygon_convex:
                                    color_validation = object_validation.validate_color(test_box, test_scene, used_src_pts,
                                                                                        used_dst_pts, bounds, homography)
                                    if color_validation:

                                        if object_validation.check_rectangularity(bounds):
                                            M = cv2.moments(bounds)
                                            cx = int(M['m10']/M['m00'])
                                            cy = int(M['m01']/M['m00'])
                                            _, _, w, _ = cv2.boundingRect(bounds)
                                            new_barycenter_mask = np.zeros(matches_mask.shape, dtype=np.uint8)
                                            cv2.circle(new_barycenter_mask, (cx, cy), w // 4, color, -1)
                                            new_bar_copy = new_barycenter_mask.copy()
                                            new_bar_copy[new_bar_copy > 0] = 255
                                            # print('Color barycenter {} = {}'.format(len(bounds_dict), color))
                                            intersection = cv2.bitwise_and(new_bar_copy, matches_mask)
                                            # visualization.display_img(new_barycenter_mask)
                                            if cv2.countNonZero(intersection) > 0:
                                                # visualization.display_img(intersection, title='Intersection')
                                                gray_index = intersection[intersection > 0][0]
                                                # print('Color intersection ', gray_index)
                                                bar_intersected = bounds_dict[gray_index]
                                                bar_intersecting = (bounds, test_box, box_name)
                                                result = object_validation.compare_detections(bar_intersected, bar_intersecting)
                                                # if result == 1 allora l'intersecato è interno all'intersecante, non faccio nulla
                                                if result == 0:   # devo sostituire nel dict e nella mask l'intersecato con l'intersecante
                                                    # per cancellare dalla mask l'intersecato disegno sul suo baricentro un baricentro nero delle stesse dimensioni
                                                    M_intersected = cv2.moments(bar_intersected[0])
                                                    cx_intersected = int(M_intersected['m10'] / M_intersected['m00'])
                                                    cy_intersected = int(M_intersected['m01'] / M_intersected['m00'])
                                                    _, _, w_intersected, _ = cv2.boundingRect(bar_intersected[0])
                                                    cv2.circle(matches_mask, (cx_intersected, cy_intersected), w_intersected // 4, (0, 0, 0), -1)

                                                    cv2.circle(matches_mask, (cx, cy), w // 4, int(gray_index), -1)
                                                    bounds_dict[gray_index] = bar_intersecting
                                                if result == -1:   # c'è intersezione ma i box non sono uno dentro l'altro, aggiungo l'intersecante al dict e alla mask
                                                    cv2.circle(matches_mask, (cx, cy), w // 4, color, -1)
                                                    bounds_dict[color] = bar_intersecting
                                                    color += 1
                                            else:   # niente intersezione, aggiungo al dict e alla mask
                                                cv2.circle(matches_mask, (cx, cy), w // 4, color, -1)
                                                bounds_dict[color] = (bounds, test_box, box_name)
                                                color += 1
                                    else:
                                        print('Box {} failed color validation'.format(box_name))
                                else:
                                    print('Box {} failed convex validation'.format(box_name))
            for key in bounds_dict:
                visualization_scene = visualization.draw_polygons(visualization_scene, [bounds_dict[key][0]])
                visualization_scene = visualization.draw_names(visualization_scene, bounds_dict[key][0], bounds_dict[key][2])

            visualization.display_img(visualization_scene)

    elif arg == '-h':
        precompute_box_features_hard()
        for scene in scenes:
            visualization_scene = scene.copy()
            visualization_scene = image_processing.resize_img(visualization_scene, 4)

            sub_scenes = parallel_hough.split_shelves(scene)

            results = parallel_hough.hough_sub_images(sub_scenes, dict_box_features)
            for row in results:
                for name in row:
                    for bounds in row[name]:
                        visualization_scene = visualization.draw_polygons(visualization_scene, [bounds])
                        #visualization_scene = visualization.draw_names(visualization_scene, bounds, name)

            visualization.display_img(visualization_scene)


main()
