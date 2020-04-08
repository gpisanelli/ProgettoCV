import cv2

import feature_matching
import result_presentation
from utils import visualization, load_images, image_processing
import object_validation


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
        proc_scene = preprocess_scene(scene)
        test_scene = scene.copy()
        test_scene = image_processing.resize_img_dim(test_scene, proc_scene.shape[1], proc_scene.shape[0])
        visualization_scene = test_scene.copy()

        kp_scene, des_scene = feature_matching.detect_features_SIFT(proc_scene)

        total_results = {}
        for name in box_names:
            total_results[name] = []

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
                success, bounds, homography, used_box_points, used_scene_points, not_used_matches = \
                    feature_matching.find_object(matches, kp_box, kp_scene, proc_box)

                if success:
                    # Object validation
                    polygon_convex = object_validation.is_convex_polygon(bounds)

                    if polygon_convex:
                        color_validation = object_validation.validate_color(test_box, test_scene, used_box_points,
                                                                            used_scene_points, bounds, homography)
                        if color_validation:
                            rectangular = object_validation.check_rectangularity(bounds, threshold=0.85)

                            if rectangular:
                                searching = False
                                total_results[box_name].append(cv2.boundingRect(bounds))

                        matches = not_used_matches
                    else:   # Polygon not convex
                        searching = False
                else:   # No homography found
                    searching = False

        result_presentation.display_result(total_results, visualization_scene)

    return total_results
