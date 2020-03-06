import time

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
    #'h1.jpg','h2.jpg','h3.jpg','h4.jpg','h5.jpg'
]
box_names = ['0.jpg', '1.jpg', '11.jpg', '19.jpg', '24.jpg', '25.jpg', '26.jpg']


def preprocess_box(b):
    pr_box = b.copy()
    # Preprocessing (box image)
    pr_box = image_processing.convert_grayscale(pr_box)
    pr_box = image_processing.equalize_histogram(pr_box)
    #pr_box = image_processing.sharpen_img(pr_box)

    return pr_box


def preprocess_scene(s):
    # Preprocessing (scene image)
    pr_scene = s.copy()
    pr_scene = image_processing.convert_grayscale(pr_scene)
    pr_scene = image_processing.equalize_histogram(pr_scene)
    pr_scene = image_processing.resize_img(pr_scene, scale_factor=2)
    pr_scene = image_processing.sharpen_img(pr_scene)

    return pr_scene




def main():
    # Since in a realistic scenario we should already already have the images of the boxes, we can precompute their features
    # and save them in a dictionary (or better initialize the dictionary from a file, which will be loaded on application
    # startup). The features will later be retrieved from the dictionary, avoiding the cost of computing them during
    # execution.
    dict_box_features = {}
    for box_name in box_names:
        box_path = load_images.get_path_for_box(box_name)
        box = load_images.load_img_color(box_path)
        proc_box = preprocess_box(box)
        # Feature detection
        kp_box, des_box = feature_detection.detect_features_SIFT(proc_box)
        dict_box_features[box_name] = (kp_box, des_box)


    for scene_name in scene_names:
        scene_path = load_images.get_path_for_scene(scene_name)
        scene = load_images.load_img_color(scene_path)
        #visualization.display_img(scene, 400, 'Scene (press Esc to continue)')

        proc_scene = preprocess_scene(scene)
        test_scene = scene.copy()
        test_scene = image_processing.resize_img_dim(test_scene, proc_scene.shape[1], proc_scene.shape[0])
        visualization_scene = test_scene.copy()
        s = time.time()
        kp_scene, des_scene = feature_detection.detect_features_SIFT(proc_scene)
        print('\nTIME DETECTION: ', time.time() - s, '\n')

        s = time.time()
        for box_name in box_names:
            #print('\n\n-----------------------------------------------------\n\n', 'Working with scene {} and box {}'.format(scene_name, box_name))

            box_path = load_images.get_path_for_box(box_name)
            box = load_images.load_img_color(box_path)
            proc_box = preprocess_box(box)
            test_box = box.copy()
            test_box = image_processing.resize_img_dim(test_box, proc_box.shape[1], proc_box.shape[0])
            #visualization.display_img(box, 200, 'Box (press Esc to continue)')

            # Feature detection
            #kp_box, des_box = feature_detection.detect_features_SIFT(proc_box)
            kp_box, des_box = dict_box_features[box_name]

            # Feature matching
            matches = feature_matching.find_matches(des_box, des_scene)
            #print('Number of matches: ', len(matches))

            if len(matches) > 10:
                # Object detection
                bounds, homography, used_box_points, used_scene_points = feature_matching.find_object(matches, kp_box, kp_scene, proc_box)

                # Object validation
                polygon_convex = object_validation.is_convex_polygon(bounds)

                if polygon_convex:
                    color_validation = object_validation.validate_color(test_box, test_scene, used_box_points, used_scene_points, bounds, homography)
                    if not color_validation:
                        print('Color validation failed')
                    else:
                        print('Color validation successful')
                        visualization_scene = visualization.draw_polygons(visualization_scene, [bounds])
                        visualization_scene = visualization.draw_names(visualization_scene, bounds, box_name)
            else:
                print('Not enough matches')
                pass

        visualization.display_img(visualization_scene, 1000, 'Result (press Esc to continue)')


main()