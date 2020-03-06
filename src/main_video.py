import time
import cv2
import numpy as np

import feature_detection
import feature_matching
import image_processing
import load_images
import object_validation
import visualization

# Image loading
template_files = [
    '../images/pixel.jpg'
]

dict_features_templates = {}


def preprocess_template(template):
    # Preprocessing (box image)
    pr_template = image_processing.convert_grayscale(template)
    pr_template = image_processing.equalize_histogram(pr_template)
    pr_template = image_processing.sharpen_img(pr_template)
    return pr_template


def preprocess_scene(frame):
    pr_scene = image_processing.convert_grayscale(frame)
    pr_scene = image_processing.equalize_histogram(pr_scene)
    pr_scene = image_processing.sharpen_img(pr_scene)
    return pr_scene



def pre_compute_templates():
    for file_name in template_files:
        template = load_images.load_img_color(file_name)
        proc_template = preprocess_template(template)
        kp_template, des_template = feature_detection.detect_features_SIFT(proc_template)
        dict_features_templates[file_name] = (template, kp_template, des_template)



def run():
    pre_compute_templates()

    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
    }

    tracker = OPENCV_OBJECT_TRACKERS['medianflow']()

    tracking_bounds = None
    found = False
    last = 0
    cap = cv2.VideoCapture(0)
    print('STARTED')

    last_search = 0

    while True:
        ret, frame = cap.read()

        if tracking_bounds is not None:
            found, tracking_bounds = tracker.update(frame)
            # Draw bounding box
            if found:
                # Tracking success
                p1 = (int(tracking_bounds[0]), int(tracking_bounds[1]))
                p2 = (int(tracking_bounds[0] + tracking_bounds[2]), int(tracking_bounds[1] + tracking_bounds[3]))
                cv2.rectangle(frame, p1, p2, (255,0,0), 5)
                #c = (int(tracking_bounds[0] + tracking_bounds[2]/2), int(tracking_bounds[1] + tracking_bounds[3]/2))
                #cv2.circle(frame, c, 5, (0,255,0), 3)
                cv2.imshow('Video', frame)
            else:
                tracker = OPENCV_OBJECT_TRACKERS['medianflow']()
                tracking_bounds = None
                cv2.putText(frame, "Tracking failure detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                cv2.imshow('Video', frame)
        else:
            cv2.putText(frame, "Tracking failure detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv2.imshow('Video', frame)

        key = cv2.waitKey(1)
        if key == 27:
            break
        if (key == ord('c') or not found) and time.time() - last_search > 0.5:
            last_search = time.time()
            last = time.time()

            pr_scene = preprocess_scene(frame)
            test_scene = frame.copy()
            #test_scene = cv2.fastNlMeansDenoisingColored(test_scene, None, 10, 10, 7, 21)
            s = time.time()
            kp_scene, des_scene = feature_detection.detect_features_SIFT(pr_scene)
            #print('TIME DETECTION: ', time.time() - s, '\n')
            for template_file in template_files:
                template, kp_template, des_template = dict_features_templates[template_file]
                test_template = template.copy()
                proc_template = preprocess_template(template)

                # Feature matching
                matches = feature_matching.find_matches(des_template, des_scene)
                if len(matches) > 10:
                    # Object detection
                    bounds, homography, used_template_points, used_scene_points = feature_matching.find_object(matches, kp_template, kp_scene, proc_template)

                    # Object validation
                    polygon_convex = object_validation.is_convex_polygon(bounds)

                    if polygon_convex:
                        #print('Convex valid')
                        color_validation = object_validation.validate_color(test_template, test_scene, used_template_points, used_scene_points, bounds, homography)

                        if color_validation:
                            #print('Color valid')

                            left = np.min(
                                [bounds[0][0], bounds[1][0], bounds[2][0], bounds[3][0]])
                            right = np.max(
                                [bounds[0][0], bounds[1][0], bounds[2][0], bounds[3][0]])
                            top = np.min(
                                [bounds[0][1], bounds[1][1], bounds[2][1], bounds[3][1]])
                            bottom = np.max(
                                [bounds[0][1], bounds[1][1], bounds[2][1], bounds[3][1]])

                            tracker = OPENCV_OBJECT_TRACKERS['medianflow']()
                            tracking_bounds = (left, top, right - left, bottom - top)
                            tracker.init(frame, tracking_bounds)
                            print('OBJECT DETECTED')

    cap.release()
    cv2.destroyAllWindows()

run()
