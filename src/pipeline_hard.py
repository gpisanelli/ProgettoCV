import cv2

import feature_matching
import parallel_hough
import result_presentation
from utils import visualization, load_images, image_processing


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
    if pr_box.shape[1] >= 200:
        pr_box = image_processing.blur_image(pr_box)
    return pr_box


def start(box_names, scenes):
    dict_box_features = precompute_box_features(box_names)

    scaling_factor = 2

    for scene in scenes:
        results = parallel_hough.hough_sub_images(scene, dict_box_features, scaling_factor)
        visualization_scene = scene.copy()

        total_results = {}
        for name in box_names:
            total_results[name] = []

        for row in results:
            for name in row:
                for b in row[name]:
                    x, y, w, h = cv2.boundingRect(b)
                    x //= scaling_factor
                    y //= scaling_factor
                    w //= scaling_factor
                    h //= scaling_factor
                    total_results[name].append((x, y, w, h))

        result_presentation.display_result(total_results, visualization_scene, draw_names=False)

    return total_results
