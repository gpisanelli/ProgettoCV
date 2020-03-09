import time

import cv2
import numpy as np

import feature_detection
import feature_matching
import image_processing
import load_images
import object_validation
import visualization

# Image loading
scene_names = [
    'e1.png','e2.png','e3.png','e4.png','e5.png'
    #'m1.png','m2.png','m3.png','m4.png','m5.png'
    #,['h' + str(n) + '.jpg' for n in range(1, 6)]
]

box_names = ['0.jpg','1.jpg','11.jpg','19.jpg','24.jpg','25.jpg','26.jpg']

dict_box_features = {}


# Since in a realistic scenario we should already already have the images of the boxes, we can precompute their features
# and save them in a dictionary (or better initialize the dictionary from a file, which will be loaded on application
# startup). The features will later be retrieved from the dictionary, avoiding the cost of computing them during
# execution.
def precompute_box_features():
    for box_name in box_names:
        box_path = load_images.get_path_for_box(box_name)
        box = load_images.load_img_color(box_path)
        proc_box = preprocess_box(box)
        kp_box, des_box = feature_detection.detect_features_SIFT(proc_box)
        dict_box_features[box_name] = (box, proc_box, kp_box, des_box)


def preprocess_box(b):
    pr_box = b.copy()
    # Preprocessing (box image)
    pr_box = image_processing.convert_grayscale(pr_box)
    pr_box = image_processing.equalize_histogram(pr_box)
    pr_box = image_processing.sharpen_img(pr_box)

    return pr_box


def preprocess_scene(s):
    # Preprocessing (scene image)
    pr_scene = s.copy()
    pr_scene = image_processing.convert_grayscale(pr_scene)
    pr_scene = image_processing.resize_img(pr_scene, 2)
    pr_scene = image_processing.equalize_histogram(pr_scene)
    pr_scene = image_processing.sharpen_img(pr_scene)

    return pr_scene


def main():
    precompute_box_features()

    for scene_name in scene_names:
        scene_path = load_images.get_path_for_scene(scene_name)
        scene = load_images.load_img_color(scene_path)
        proc_scene = preprocess_scene(scene)
        test_scene = scene.copy()
        test_scene = image_processing.resize_img_dim(test_scene, proc_scene.shape[1], proc_scene.shape[0])
        visualization_scene = test_scene.copy()

        s = time.time()
        kp_scene, des_scene = feature_detection.detect_features_SIFT(proc_scene)
        print('\nTIME DETECTION: ', time.time() - s, '\n')

        s = time.time()
        for box_name in box_names:
            # Box features retrieval
            box, proc_box, kp_box, des_box = dict_box_features[box_name]
            test_box = box.copy()
            test_box = image_processing.resize_img_dim(test_box, proc_box.shape[1], proc_box.shape[0])

            # Feature matching
            matches = feature_matching.find_matches(des_box, des_scene)

            searching = True
            while len(matches) > 10 and searching:
                # Object detection
                bounds, homography, used_box_points, used_scene_points, not_used_matches = \
                    feature_matching.find_object(matches, kp_box, kp_scene, proc_box)

                # Object validation
                polygon_convex = object_validation.is_convex_polygon(bounds)

                if polygon_convex:
                    color_validation = object_validation.validate_color(test_box, test_scene, used_box_points,
                                                                        used_scene_points, bounds, homography)
                    if color_validation:
                        searching = False
                        visualization_scene = visualization.draw_polygons(visualization_scene, [bounds])
                        visualization_scene = visualization.draw_names(visualization_scene, bounds, box_name)

                    matches = not_used_matches
                else:
                    searching = False

        visualization.display_img(visualization_scene, 800, 'Result (press Esc to continue)')


main()