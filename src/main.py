import feature_detection
import feature_matching
import image_processing
import load_images
import visualization


# Image loading
box_path = load_images.get_path_for_box('0.jpg')
scene_path = load_images.get_path_for_scene('h1.jpg')

box = load_images.load_img_grayscale(box_path)
scene = load_images.load_img_grayscale(scene_path)
visualization_scene = load_images.load_img_color(scene_path)


# Preprocessing (box image)
proc_box = image_processing.equalize_histogram(box)
proc_box = image_processing.sharpen_img(proc_box)

# Preprocessing (scene image)
proc_scene = image_processing.equalize_histogram(scene)
proc_scene = image_processing.resize_img(proc_scene, 2)
proc_scene = image_processing.sharpen_img(proc_scene)


# Feature detection
kp_box, des_box = feature_detection.detect_features_SIFT(proc_box)
kp_scene, des_scene = feature_detection.detect_features_SIFT(proc_scene)

# Feature matching
matches = feature_matching.find_matches(des_box, des_scene)
print('Matches: ', len(matches))

# Object detection
bounds, homography = feature_matching.find_object(matches, kp_box, kp_scene, proc_box)

# Result visualization
visualization_scene = image_processing.resize_img_dim(visualization_scene, proc_scene.shape[1], proc_scene.shape[0])
result = visualization.draw_polygons(visualization_scene, [bounds])
visualization.display_img(result, 1000, 'Result')
