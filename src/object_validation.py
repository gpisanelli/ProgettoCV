import time

import cv2
import numpy as np

import feature_matching
import image_processing
import visualization
import matplotlib.pyplot as plt
import math

HISTOGRAM_THRESHOLD = 1.5


def is_convex_polygon(bounds):
    return cv2.isContourConvex(bounds)


def compute_rectangularity(bounds):
    x, y, w, h = cv2.boundingRect(bounds)
    bounds_area = cv2.contourArea(bounds)
    rect_area = w * h
    return bounds_area / rect_area


def check_rectangularity(bounds):
    return compute_rectangularity(bounds) > 0.85


def is_contained(bounds_outer, bounds_inner):
    x_o, y_o, w_o, h_o = cv2.boundingRect(bounds_outer)
    x_i, y_i, w_i, h_i = cv2.boundingRect(bounds_inner)
    ul = (x_o, y_o) <= (x_i, y_i)
    bl = x_o <= x_i and y_o + h_o >= y_i + h_i
    br = (x_o + w_o, y_o + h_o) >= (x_i + w_i, y_i + h_i)
    ur = x_o + w_o >= x_i + w_i and y_o <= y_i

    return bl and ul and ur and br


def compare_detections(intersected, intersecting):
    contained2in1 = is_contained(intersected[0], intersecting[0])
    contained1in2 = is_contained(intersecting[0], intersected[0])
    if contained2in1:  # intersecting e' completamente interno o uguale a intersected
        return 1
    elif contained1in2:  # intersected e' completamente interno o uguale a intersecting
        return 0
    else:   # c'è intersezione ma i box non sono l'uno dentro l'altro
        return -1


def is_contained_hard(bounds_outer, bounds_inner, outer_scaling):
    x_o, y_o, w_o, h_o = cv2.boundingRect(bounds_outer)
    x_i, y_i, w_i, h_i = cv2.boundingRect(bounds_inner)

    # con le operazioni successive scalo (x_o, y_o), w_o, h_o in modo che definiscano un rettangolo con area
    # ingrandita in base ad outer_scaling (ad esempio con 1.5 aumenta del 50%)
    middle_x, middle_y = x_o + (w_o / 2), y_o + (h_o / 2)
    w_o, h_o = w_o * math.sqrt(outer_scaling), h_o * math.sqrt(outer_scaling)
    x_o, y_o = middle_x - (w_o / 2), middle_y - (h_o / 2)

    ul = (x_o, y_o) <= (x_i, y_i)
    bl = x_o <= x_i and y_o + h_o >= y_i + h_i
    br = (x_o + w_o, y_o + h_o) >= (x_i + w_i, y_i + h_i)
    ur = x_o + w_o >= x_i + w_i and y_o <= y_i

    return bl and ul and ur and br


def find_best(bounds1, bounds2):
    rectangularity1 = compute_rectangularity(bounds1)
    rectangularity2 = compute_rectangularity(bounds2)
    if rectangularity1 >= rectangularity2:
        return 1
    else:
        return 0


# ritorna 1 se devo mantenere intersected, 0 se devo mantenere intersecting, -1 se devo mantenere entrambi
def compare_detections_hard(intersected, intersecting, outer_scaling=1):
    contained2in1 = is_contained_hard(intersected[0], intersecting[0], outer_scaling)
    contained1in2 = is_contained_hard(intersecting[0], intersected[0], outer_scaling)  # probabilmente useless, i box dovrebbero essere così simili che è indifferente quale ingrandisco
    if contained2in1 or contained1in2:  # due box sono troppo sovrapposti, mantengo solo il migliore
        return find_best(intersected[0], intersecting[0])
    else:  # c'è intersezione ma i box non sono troppo sovrapposti
        return -1


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

    t = box_masked_area[box_masked_area > 0]
    area_ratio = t.shape[0] / (masked_box.shape[0] * masked_box.shape[1])

    if area_ratio < 0.20:
        return False

    return compare_hue(masked_box, masked_scene, homography, match_bounds)


# Compares the colors of the template with the instance found in the scene, to test the compatibility of the match. This
# is done by computing the histogram of the hue channel of the two images, grouping the values into 8 bins, in order to
# group together similar colors. Then, the correlation of the two histograms is computed, as well as their peaks, which
# indicate the main colors of the images. Matches with a low correlation or without enough peaks in common are
# discarded.
def compare_hue(box, scene, homography, match_bounds):
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
    hist1 = cv2.calcHist([h1], channels=[0], mask=mask1, histSize=[8], ranges=[0, 180])
    hist2 = cv2.calcHist([h2], channels=[0], mask=mask1, histSize=[8], ranges=[0, 180])

    # Shift red values from the end of the range to the start. This is done because hue is represented as a circle
    # with values [0-180] in opencv, and red values are present both in the range [0-30] and in the range [150-180]
    # approximately.
    hist1[0] = hist1[0] + hist1[7]
    hist2[0] = hist2[0] + hist2[7]
    hist1[7] = 0
    hist2[7] = 0

    cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    #plt.plot(hist1, color='#FF4455')
    #plt.plot(hist2)
    #plt.show()

    # Compare the two hue histograms using correlation
    hue_comparison = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

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

    return common_peaks >= 2 and hue_comparison > 0.8