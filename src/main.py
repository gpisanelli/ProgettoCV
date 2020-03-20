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
def precompute_box_features():
    for box_name in box_dict['-h']:
        box_path = load_images.get_path_for_box(box_name)
        box = load_images.load_img_color(box_path)
        proc_box = preprocess_box(box)
        kp_box, des_box = feature_detection.detect_features_SIFT(proc_box)
        dict_box_features[box_name] = (box, proc_box, kp_box, des_box)


def preprocess_box(b):
    pr_box = b.copy()
    pr_box = image_processing.convert_grayscale(pr_box)
    pr_box = image_processing.equalize_histogram(pr_box)
    if pr_box.shape[0] >= 300:
        pr_box = image_processing.blur_image(pr_box)

    return pr_box


def preprocess_scene(s):
    # Preprocessing (scene image)
    pr_scene = s.copy()
    pr_scene = image_processing.convert_grayscale(pr_scene)
    pr_scene = image_processing.equalize_histogram(pr_scene)
    pr_scene = image_processing.sharpen_img(pr_scene)

    return pr_scene


def main():
    accepted_list = ['-e', '-m', '-h']
    arg = sys.argv[1]
    if len(sys.argv) != 2 or arg not in accepted_list:
        print('usage: ./main.py -(e|h|m)')
        sys.exit(2)

    precompute_box_features()

    scene_names = scene_dict[arg]
    scenes = []

    for scene_name in scene_names:
        scene_path = load_images.get_path_for_scene(scene_name)
        scenes.append(load_images.load_img_color(scene_path))

    if arg == '-e':
        for scene in scenes:
            proc_scene = preprocess_scene(scene)
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
        for scene in scenes:
            # forse preprocessing to change per le medium
            # template_color = load_images.load_img_color(template_path)
            # template = image_processing.convert_grayscale(template_color)
            # template = image_processing.equalize_histogram(template)

            scene_color = scene.copy()
            scene = preprocess_scene(scene)
            test_scene = scene_color.copy()
            test_scene = image_processing.resize_img_dim(test_scene, scene.shape[1], scene.shape[0])
            visualization_scene = test_scene.copy()  # necessarie tutte ste scene?

            # s = time.time()
            kp_s, des_s = feature_detection.detect_features_SIFT(scene)
            # print('\nTIME DETECTION: ', time.time() - s, '\n')

            matches_mask = np.zeros(scene.shape, dtype=np.uint8)

            color = 0
            matches_dict = {}

            for box_name in box_dict['-m']:
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
                        if len(good_matches[i][1]) >= 6:
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
                                        x, y, w, h = cv2.boundingRect(bounds)

                                        if cv2.contourArea(bounds) / (w * h) >= 0.9:
                                            visualization_scene = visualization.draw_polygons(visualization_scene, [bounds])
                                            visualization_scene = visualization.draw_names(visualization_scene, bounds, box_name)

                                            M = cv2.moments(bounds)
                                            cx = int(M['m10']/M['m00'])
                                            cy = int(M['m01']/M['m00'])
                                            _, _, w, _ = cv2.boundingRect(bounds)
                                            new_barycenter_mask = np.zeros(matches_mask.shape, dtype=np.uint8)
                                            cv2.circle(new_barycenter_mask, (cx, cy), w // 4, color + 10, -1)
                                            intersection = cv2.bitwise_and(new_barycenter_mask, matches_mask)
                                            visualization.display_img(new_barycenter_mask)
                                            if cv2.countNonZero(intersection) > 0:
                                                visualization.display_img(intersection)
                                                gray_index = matches_mask[cv2.findNonZero(intersection)[0][0][0], cv2.findNonZero(intersection)[0][0][1]]

                                            cv2.circle(matches_mask, (cx, cy), w // 4, color + 10, -1)

                                            matches_dict[color] = (bounds, test_box)
                                    else:
                                        print('Box {} failed color validation'.format(box_name))
                                else:
                                    print('Box {} failed convex validation'.format(box_name))
                        color += 10
                        '''
                        caso in cui non ho abbastanza keypoint per la homography, non dovrebbe servire e anyway non funge molto bene
    
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
                            box_vertexes = [(0, 0), (proc_box.shape[1], 0), (proc_box.shape[1], proc_box.shape[0]), (0, proc_box.shape[0])]
                            scene_vertexes = []
                            vectors = []
                            int_barycenter = np.int32(barycenter)
                            for j in range(4):
                                vectors.append(np.subtract(box_vertexes[j], int_barycenter))
    
                            # print('Box vertexes {} = {}'.format(i, box_vertexes))
                            # print('Box vectors {} = {}'.format(i, vectors))
                            template_copy = proc_box.copy()
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
    
                            # print('Scene vertexes match {} = {}'.format(i, scene_vertexes))
                            scene_copy = scene.copy()
                            for j in range(4):
                                cv2.circle(scene_copy, (int(good_matches[i][0][0]), int(good_matches[i][0][1])), 10, (0, 255, 0), -1)
                                cv2.circle(scene_copy, scene_vertexes[j], 10, (0, 255, 0), -1)
                                cv2.arrowedLine(scene_copy, scene_vertexes[j], (int(round(scene_vertexes[j][0]-vectors[j][0])),
                                                                                int(round(scene_vertexes[j][1]-vectors[j][1]))), (0, 255, 0))
                            visualization.display_img(scene_copy, title='Scene_vertexes_vectors')
    
                            # Object validation
                            polygon_convex = object_validation.is_convex_polygon(scene_vertexes)
                            if polygon_convex:
                                # homography sarebbe da computare tra gli 8 punti vertici del box e della scena, gli used_points sono quei vertici
                                color_validation = object_validation.validate_color(test_box, test_scene, used_src_points,
                                                                                used_dst_points, scene_vertexes, homography)
                                if color_validation:
                                    visualization_scene = visualization.draw_polygons(visualization_scene, [scene_vertexes])
                                    visualization_scene = visualization.draw_names(visualization_scene, scene_vertexes, box_name)
                                else:
                                    print('Box {} failed color validation'.format(box_name))
                            else:
                                print('Box {} failed convex validation'.format(box_name))
    
                            visualization_scene = visualization.draw_polygons(visualization_scene, np.int32([scene_vertexes]))
                        '''
            visualization.display_img(visualization_scene)

    elif arg == '-h':
        for scene in scenes:
            scene = image_processing.resize_img(scene, 2)

            visualization_scene = scene.copy()

            sub_scenes = parallel_hough.split_shelves(scene)

            results = parallel_hough.hough_sub_images(sub_scenes, dict_box_features)
            for row in results:
                for name in row:
                    for bounds in row[name]:
                        visualization_scene = visualization.draw_polygons(visualization_scene, [bounds])
                        visualization_scene = visualization.draw_names(visualization_scene, bounds, name)

            visualization.display_img(visualization_scene)


main()