import cv2
import numpy as np

import feature_matching
import hough_detection
import result_presentation
from utils import visualization, load_images, image_processing
import object_validation

MATCHES_THRESHOLD = 6


def precompute_box_features(box_names):
    dict_box_features = {}

    for box_name in box_names:
        box_path = load_images.get_path_for_box(box_name)
        box = load_images.load_img_color(box_path)
        proc_box = preprocess_box(box)
        kp_box, des_box = feature_matching.detect_features_SIFT(proc_box)
        dict_box_features[box_name] = (box, proc_box, kp_box, des_box)

    return dict_box_features


def preprocess_box(b):
    pr_box = b.copy()
    pr_box = image_processing.convert_grayscale(pr_box)
    return pr_box


def preprocess_scene(s):
    pr_scene = s.copy()
    pr_scene = image_processing.convert_grayscale(pr_scene)
    pr_scene = image_processing.sharpen_img(pr_scene)
    return pr_scene


def start(box_names, scenes):
    dict_box_features = precompute_box_features(box_names)

    for scene in scenes:
        scene_color = scene.copy()
        proc_scene = preprocess_scene(scene_color)
        test_scene = scene_color.copy()
        test_scene = image_processing.resize_img_dim(test_scene, proc_scene.shape[1], proc_scene.shape[0])
        visualization_scene = test_scene.copy()

        kp_s, des_s = feature_matching.detect_features_SIFT(proc_scene)

        matches_mask = np.zeros(proc_scene.shape, dtype=np.uint8)

        color = 1
        bounds_dict = {}

        total_results = {}
        for name in box_names:
            total_results[name] = []

        for box_name in box_names:
            box, proc_box, kp_t, des_t = dict_box_features[box_name]

            test_box = box.copy()
            test_box = image_processing.resize_img_dim(test_box, proc_box.shape[1], proc_box.shape[0])

            matches = feature_matching.find_matches(des_t, des_s)
            good_matches = hough_detection.find_matches(matches, kp_t, kp_s, proc_scene)

            for i in good_matches:
                if len(good_matches[i][1]) >= MATCHES_THRESHOLD:
                    success, bounds, homography, used_src_pts, used_dst_pts, not_used_matches = \
                        feature_matching.find_object(good_matches[i][1], kp_t, kp_s, proc_box)

                    if success:
                        polygon_convex = object_validation.is_convex_polygon(bounds)

                        if polygon_convex:
                            color_validation = \
                                object_validation.validate_color(test_box, test_scene, used_src_pts, used_dst_pts, bounds, homography)

                            if color_validation:

                                if object_validation.check_rectangularity(bounds, threshold=0.85):

                                    matches_mask, color = object_validation \
                                        .check_matches_intersection(bounds, test_box, box_name, bounds_dict, matches_mask, color)

        for key in bounds_dict:
            box_bounds = bounds_dict[key][0]
            box_name = bounds_dict[key][2]
            total_results[box_name].append(cv2.boundingRect(box_bounds))

        result_presentation.display_result(total_results, visualization_scene)

    return total_results
