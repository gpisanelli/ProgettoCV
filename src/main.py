import cv2
import numpy as np

import feature_detection
import feature_matching
import image_processing
import load_images
import object_validation
import visualization

# Image loading
scene_names = ['e5.png']
box_names = ['0.jpg', '1.jpg', '11.jpg', '19.jpg', '24.jpg', '25.jpg', '26.jpg']


def preprocess_box(b):
    box_copy = b.copy()
    # Preprocessing (box image)
    pr_box = image_processing.convert_grayscale(box_copy)
    pr_box = image_processing.equalize_histogram(pr_box)
    pr_box = image_processing.sharpen_img(pr_box)

    return pr_box


def preprocess_scene(s):
    scene_copy = s.copy()
    # Preprocessing (scene image)
    pr_scene = image_processing.convert_grayscale(scene_copy)
    pr_scene = image_processing.equalize_histogram(pr_scene)
    pr_scene = image_processing.resize_img(pr_scene, scale_factor=2)
    pr_scene = image_processing.sharpen_img(pr_scene)

    return pr_scene


#scene_name = 'e1.png'
#box_name = '0.jpg'
for scene_name in scene_names:
    scene_path = load_images.get_path_for_scene(scene_name)
    scene = load_images.load_img_color(scene_path)
    visualization.display_img(scene, 800, 'Scene (press Esc to continue)')

    for box_name in box_names:
        print('\n\n-----------------------------------------------------\n\n',
              'Working with scene {} and box {}'.format(scene_name, box_name))

        box_path = load_images.get_path_for_box(box_name)
        box = load_images.load_img_color(box_path)

        visualization.display_img(box, 200, 'Box (press Esc to continue)')

        proc_box = preprocess_box(box)
        proc_scene = preprocess_scene(scene)
        test_box = box.copy()
        test_scene = scene.copy()
        test_scene = image_processing.resize_img_dim(test_scene, proc_scene.shape[1], proc_scene.shape[0])
        visualization_scene = test_scene.copy()

        # Feature detection
        kp_box, des_box = feature_detection.detect_features_SIFT(proc_box)
        kp_scene, des_scene = feature_detection.detect_features_SIFT(proc_scene)

        # Feature matching
        matches = feature_matching.find_matches(des_box, des_scene)
        print('Number of matches: ', len(matches))

        if len(matches) > 10:
            # Object detection
            bounds, homography, used_box_points, used_scene_points = feature_matching.find_object(matches, kp_box, kp_scene, proc_box)

            # Object validation
            polygon_convex = object_validation.is_convex_polygon(bounds)
            print('Convex polygon: ', polygon_convex)

            if polygon_convex:
                color_validation = object_validation.validate_color(test_box, test_scene, used_box_points, used_scene_points, bounds, homography)
                if not color_validation:
                    print('Color validation failed')
                else:
                    print('Color validation successful')
                    # Result visualization
                    visualization_scene = image_processing.resize_img_dim(visualization_scene, proc_scene.shape[1],
                                                                          proc_scene.shape[0])
                    result = visualization.draw_polygons(visualization_scene, [bounds])
                    result = visualization.draw_names(result, bounds, box_name)
                    visualization.display_img(result, 1000, 'Result (press Esc to continue)')
        else:
            print('Not enough matches')