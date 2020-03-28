import feature_matching
import object_validation
import parallel_hough
import prove
import image_processing
import load_images
import feature_detection
import visualization

scene_names = ['h1.jpg', 'h2.jpg', 'h3.jpg', 'h4.jpg', 'h5.jpg']
boxes = ['0.jpg', '1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6.jpg',
           '7.jpg', '8.jpg', '9.jpg', '10.jpg', '11.jpg', '12.jpg', '13.jpg',
           '14.jpg', '15.jpg', '16.jpg', '17.jpg', '18.jpg', '19.jpg', '20.jpg',
           '21.jpg', '22.jpg', '23.jpg', '24.jpg', '25.jpg', '26.jpg']

dict_box_features = {}

def precompute_box_features_hard():
    for box_name in boxes:
        box_path = load_images.get_path_for_box(box_name)
        box_path = '../'+box_path
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


precompute_box_features_hard()
scenes = []
for scene_name in scene_names:
    scene_path = load_images.get_path_for_scene(scene_name)
    scene_path = '../' + scene_path
    scenes.append(load_images.load_img_color(scene_path))

for scene in scenes:
    visualization_scene = scene.copy()
    visualization_scene = image_processing.resize_img(visualization_scene, 2)

    sub_scenes = parallel_hough.split_shelves(scene)

    results = parallel_hough.hough_sub_images(sub_scenes, dict_box_features)
    for row in results:
        for name in row:
            for bounds in row[name]:
                visualization_scene = visualization.draw_polygons(visualization_scene, [bounds])
                # visualization_scene = visualization.draw_names(visualization_scene, bounds, name)

    visualization.display_img(visualization_scene)
