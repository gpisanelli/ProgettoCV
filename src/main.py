import feature_detection
import feature_matching
import image_processing
import load_images
import object_validation
import visualization


# Image loading
box_path = load_images.get_path_for_box('0.jpg')
scene_path = load_images.get_path_for_scene('e1.png')

box = load_images.load_img_color(box_path)
scene = load_images.load_img_color(scene_path)
visualization_scene = scene.copy()

visualization.display_img(box, 200, 'Box (press Esc to continue)')
visualization.display_img(scene, 800, 'Scene (press Esc to continue)')

# Preprocessing (box image)
proc_box = image_processing.convert_grayscale(box)
proc_box = image_processing.equalize_histogram(proc_box)
proc_box = image_processing.sharpen_img(proc_box)
test_box = box.copy()

# Preprocessing (scene image)
proc_scene = image_processing.convert_grayscale(scene)
proc_scene = image_processing.equalize_histogram(proc_scene)
proc_scene = image_processing.resize_img(proc_scene, 2)
proc_scene = image_processing.sharpen_img(proc_scene)
test_scene = image_processing.resize_img(scene, 2)


# Feature detection
kp_box, des_box = feature_detection.detect_features_SIFT(proc_box)
kp_scene, des_scene = feature_detection.detect_features_SIFT(proc_scene)


# Feature matching
matches = feature_matching.find_matches(des_box, des_scene)
print('Number of matches: ', len(matches))

if len(matches) > 10:
    # Object detection
    bounds, homography = feature_matching.find_object(matches, kp_box, kp_scene, proc_box)

    # Object validation
    polygon_convex = object_validation.is_convex_polygon(bounds)
    print('Convex polygon: ', polygon_convex)
    hue_comparison = object_validation.compare_hue(test_box, test_scene, homography, bounds)
    print('Hue comparison: ', hue_comparison)

    if not hue_comparison or not polygon_convex:
        print('Matched object validation failed')
    else:
        print('Validation successful')

    # Result visualization
    visualization_scene = image_processing.resize_img_dim(visualization_scene, proc_scene.shape[1], proc_scene.shape[0])
    result = visualization.draw_polygons(visualization_scene, [bounds])
    visualization.display_img(result, 1000, 'Result (press Esc to continue)')
else:
    print('Not enough matches')
