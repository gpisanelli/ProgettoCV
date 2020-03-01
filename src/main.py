import load_images
import visualization
import feature_detection
import feature_matching


box = load_images.load_img_color('../images/object_detection_project/models/0.jpg')
scene = load_images.load_img_color('../images/object_detection_project/scenes/h5.jpg')

kp1, des1 = feature_detection.detect_features_SIFT(box)
kp2, des2 = feature_detection.detect_features_SIFT(scene)

matches = feature_matching.find_matches(des1, des2)

bounds, homography = feature_matching.find_object(matches, kp1, kp2, scene)

print('Matches: ', len(matches))
print('Bounds: ', len(bounds))

result = visualization.draw_polygons(scene, [bounds])
visualization.display_img(result, 1000, 'Result')