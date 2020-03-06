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


# Draws a black rectangle around each matching point found in the two images, obtaining a box and a scene in which only
# the regions of the image that contain no important features are left. Because of how the feature detection algorithm
# works, features belong to areas with edges and intensity variations. In the case of grocery products, regions with
# many features are represented by drawings, logos, mascots and product names, which can be equal in different variants
# of the same product, thus producing wrong matches. Removing these regions from the color comparison step gives more
# reliable results, reducing the number of false positives.
# Moreover, an approximation of the percentage of the area of the image covered by features is computed. If this
# percentage is lower than a certain threshold, it means that only a small area of the image was matched (this typically
# happens when logos are matched between two different products of the same brand), and the result can be discarded to
# be sure to avoid a false positive.
# The evaluation process is very efficient, and does not have a significant impact on execution time.
def validate_color(box, scene, used_box_pts, used_scene_pts, match_bounds, homography):
    start = time.time()
    box_val = box.copy()
    scene_val = scene.copy()

    left = np.min([match_bounds[0][0], match_bounds[1][0], match_bounds[2][0], match_bounds[3][0]])
    right = np.max([match_bounds[0][0], match_bounds[1][0], match_bounds[2][0], match_bounds[3][0]])
    top = np.min([match_bounds[0][1], match_bounds[1][1], match_bounds[2][1], match_bounds[3][1]])
    bottom = np.max([match_bounds[0][1], match_bounds[1][1], match_bounds[2][1], match_bounds[3][1]])

    width = right - left
    height = bottom - top

    ratio = box_val.shape[1] / width

    # Draw mask of matches for scene image
    rect_size = min(width/20, height/20)
    masked_scene = scene_val.copy()
    for point in used_scene_pts:
        masked_scene[int(point[1] - rect_size):int(point[1] + rect_size), int(point[0] - rect_size):int(point[0] + rect_size)] = 0

    # Draw mask of matches for box image
    rect_size = rect_size * ratio
    masked_box = box_val.copy()
    box_masked_area = np.zeros((masked_box.shape[0], masked_box.shape[1]), dtype=np.uint8)
    for point in used_box_pts:
        masked_box[int(point[1] - rect_size):int(point[1] + rect_size),
        int(point[0] - rect_size):int(point[0] + rect_size)] = 0
        box_masked_area[int(point[1] - rect_size*2):int(point[1] + rect_size*2), int(point[0] - rect_size*2):int(point[0] + rect_size*2)] = 1

    #visualization.display_img(box_masked_area*255, title='Area', wait=False)
    #visualization.display_img(masked_box, 400, wait=False)
    #visualization.display_img(masked_scene, 800)
    t = box_masked_area[box_masked_area > 0]
    area_ratio = t.shape[0] / (masked_box.shape[0] * masked_box.shape[1])
    #print('Area ratio: ', area_ratio)

    if area_ratio < 0.20:
        return False

    comp_hue = compare_hue(masked_box, masked_scene, homography, match_bounds)
    print(comp_hue)
    #print('\n\nTIME: ', (time.time() - start), '\n\n')
    return comp_hue


def compare_hue(box, scene, homography, match_bounds):
    start = time.time()
    transformed_box, test_scene = image_processing.transform_box_in_scene(box, scene, homography)

    left = max(0, np.min([match_bounds[0][0], match_bounds[1][0], match_bounds[2][0], match_bounds[3][0]]))
    right = min(transformed_box.shape[1], np.max([match_bounds[0][0], match_bounds[1][0], match_bounds[2][0], match_bounds[3][0]]))
    top = max(0, np.min([match_bounds[0][1], match_bounds[1][1], match_bounds[2][1], match_bounds[3][1]]))
    bottom = min(transformed_box.shape[0], np.max([match_bounds[0][1], match_bounds[1][1], match_bounds[2][1], match_bounds[3][1]]))

    img1 = transformed_box[top:bottom, left:right]
    img2 = test_scene[top:bottom, left:right]

    h1, s1, v1 = cv2.split(cv2.cvtColor(img1, cv2.COLOR_BGR2HSV))
    h2, s2, v2 = cv2.split(cv2.cvtColor(img2, cv2.COLOR_BGR2HSV))

    mask1 = v1.copy()
    mask2 = v2.copy()
    mask1[v1 > 0] = 1
    mask2[v2 > 0] = 1

    # Hue histogram calculation, using 8 bins to group together similar colors, even though they are not identical.
    hist1 = cv2.calcHist([h1], channels=[0], mask=mask1, histSize=[6], ranges=[0, 180])
    hist2 = cv2.calcHist([h2], channels=[0], mask=mask1, histSize=[6], ranges=[0, 180])

    # Shift red values from the end of the range to the start. This is done because hue is represented as a circle
    # with values [0-180] in opencv, and red values are present both in the range [0-30] and in the range [150-180]
    # approximately.
    hist1[0] = hist1[0] + hist1[5]
    hist2[0] = hist2[0] + hist2[5]
    hist1[5] = 0
    hist2[5] = 0

    cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    #plt.plot(hist1, color='#FF4455')
    #plt.plot(hist2)
    #plt.show()

    # Compare the two hue histograms using correlation
    hue_comparison = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    print('Hue comparison: ', hue_comparison)
    if hue_comparison > 0.9:
        return True


    # Search for peaks in the hue histograms. This is done to find the dominant colors in the two images. The match is
    # considered correct if a sufficient number of common peaks are found in the two histograms, indicating similar
    # dominant colors for the two images.
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

    common_peaks = 0
    for peak in peaks2:
        if np.isin(peak, peaks1):
            common_peaks = common_peaks + 1

    print('Common peaks: ', common_peaks)

    #start = time.time()
    """
    scene_h = cv2.split(cv2.cvtColor(scene, cv2.COLOR_BGR2HSV))[0]
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
    #print('\n\nTime: ', t, '\n\n')

    test_intersection_mask = np.zeros((scene.shape[0],scene.shape[1]))
    test_intersection_mask[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = 1
    test_img = np.zeros((scene.shape[0],scene.shape[1]))
    test_img[top:bottom, left:right] = 1
    test_img[test_intersection_mask == 0] = 0
    intersection_area = cv2.countNonZero(test_img)

    rectangle_area = w * h
    intersection_percentage = intersection_area / rectangle_area
    #print('Intersection percentage: ', intersection_percentage)

    if intersection_percentage < 0.5:
        #print('Color validation failed')
        return False
    """
    if common_peaks >= 3 and hue_comparison > 0.6:
        return True
    return common_peaks >= 2 and hue_comparison > 0.75
