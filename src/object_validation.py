import time

import cv2
import numpy as np

import feature_matching
import image_processing
import visualization
import matplotlib.pyplot as plt

HISTOGRAM_THRESHOLD = 1.5


def is_convex_polygon(bounds):
    return cv2.isContourConvex(bounds)


def compare_hue(box, scene, homography, match_bounds):
    start = time.time()
    transformed_box, test_scene = image_processing.transform_box_in_scene(box, scene, homography)

    left = max(0, np.min([match_bounds[0][0], match_bounds[1][0], match_bounds[2][0], match_bounds[3][0]]))
    right = min(transformed_box.shape[1], np.max([match_bounds[0][0], match_bounds[1][0], match_bounds[2][0], match_bounds[3][0]]))
    top = max(0, np.min([match_bounds[0][1], match_bounds[1][1], match_bounds[2][1], match_bounds[3][1]]))
    bottom = min(transformed_box.shape[0], np.max([match_bounds[0][1], match_bounds[1][1], match_bounds[2][1], match_bounds[3][1]]))

    print('Left = ', left)
    print('Right = ', right)
    print('Top = ', top)
    print('Bottom = ', bottom)

    img1 = transformed_box[top:bottom, left:right]
    img2 = test_scene[top:bottom, left:right]

    #img1 = image_processing.resize_img(img1, 2)
    #img2 = image_processing.resize_img(img2, 2)

    h1, s1, v1 = cv2.split(cv2.cvtColor(img1, cv2.COLOR_BGR2HSV))
    h2, s2, v2 = cv2.split(cv2.cvtColor(img2, cv2.COLOR_BGR2HSV))

    mask1 = v1.copy()
    mask2 = v2.copy()
    mask1[v1 > 0] = 1
    mask2[v2 > 0] = 1

    # Hue comparison
    hist1 = cv2.calcHist([h1], channels=[0], mask=mask1, histSize=[8], ranges=[0, 180])
    hist2 = cv2.calcHist([h2], channels=[0], mask=mask1, histSize=[8], ranges=[0, 180])

    cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    #plt.plot(hist1)
    #plt.plot(hist2)
    #plt.show()

    hue_comparison = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    print('Hue comparison: ', hue_comparison)
    if hue_comparison > 0.9:
        return True

    _, _, _, peak1_hist1 = cv2.minMaxLoc(hist1)
    hist1[peak1_hist1[1], 0] = 0
    _, _, _, peak2_hist1 = cv2.minMaxLoc(hist1)
    hist1[peak2_hist1[1], 0] = 0
    _, _, _, peak3_hist1 = cv2.minMaxLoc(hist1)
    hist1[peak3_hist1[1], 0] = 0

    _, _, _, peak1_hist2 = cv2.minMaxLoc(hist2)
    hist2[peak1_hist2[1], 0] = 0
    _, _, _, peak2_hist2 = cv2.minMaxLoc(hist2)
    hist2[peak2_hist2[1], 0] = 0
    _, _, _, peak3_hist2 = cv2.minMaxLoc(hist2)
    hist2[peak3_hist2[1], 0] = 0

    peaks1 = list({peak1_hist1[1], peak2_hist1[1], peak3_hist1[1]})
    peaks2 = list({peak1_hist2[1], peak2_hist2[1], peak3_hist2[1]})

    print('Peaks box: ', peaks1)
    print('Peaks scene: ', peaks2)

    common_peaks = 0
    for peak in peaks2:
        if np.isin(peak, peaks1):
            common_peaks = common_peaks + 1

    print('Common peaks: ', common_peaks)

    #start = time.time()

    scene_h = cv2.split(cv2.cvtColor(test_scene, cv2.COLOR_BGR2HSV))[0]
    img1_h = cv2.split(cv2.cvtColor(img1, cv2.COLOR_BGR2HSV))[0]

    result = cv2.matchTemplate(scene_h, img1_h, cv2.TM_SQDIFF_NORMED)
    result = cv2.normalize(result, result, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    w = img1.shape[1]
    h = img1.shape[0]

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    #if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
    top_left = min_loc
    #else:
    #top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(scene, top_left, bottom_right, 255, 10)
    visualization.display_img(scene)
    t = time.time() - start
    print('\n\nTime: ', t, '\n\n')

    test_intersection_mask = np.zeros((scene.shape[0],scene.shape[1]))
    test_intersection_mask[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = 1
    test_img = np.zeros((scene.shape[0],scene.shape[1]))
    test_img[top:bottom, left:right] = 1
    test_img[test_intersection_mask == 0] = 0
    intersection_area = cv2.countNonZero(test_img)

    rectangle_area = w * h
    intersection_percentage = intersection_area / rectangle_area
    print('Intersection percentage: ', intersection_percentage)

    if intersection_percentage < 0.5:
        print('Color validation failed')
        return False

    return common_peaks >= 2
