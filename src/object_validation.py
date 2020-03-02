import cv2
import numpy as np
import image_processing
import visualization
import matplotlib.pyplot as plt

HISTOGRAM_THRESHOLD = 1.5


def is_convex_polygon(bounds):
    return cv2.isContourConvex(bounds)


def compare_hue(box, scene, homography, match_bounds):
    transformed_box, test_scene = image_processing.transform_box_in_scene(box, scene, homography)

    left = np.min([match_bounds[0][0], match_bounds[1][0], match_bounds[2][0], match_bounds[3][0]])
    right = np.max([match_bounds[0][0], match_bounds[1][0], match_bounds[2][0], match_bounds[3][0]])
    top = np.min([match_bounds[0][1], match_bounds[1][1], match_bounds[2][1], match_bounds[3][1]])
    bottom = np.max([match_bounds[0][1], match_bounds[1][1], match_bounds[2][1], match_bounds[3][1]])

    if left < 0:
        left = 0
    if top < 0:
        top = 0
    if right > transformed_box.shape[1]:
        right = transformed_box.shape[1]
    if bottom > transformed_box.shape[0]:
        bottom = transformed_box.shape[0]

    img1 = transformed_box[top:bottom, left:right]
    img2 = test_scene[top:bottom, left:right]

    img1 = image_processing.resize_img(img1, 2)
    img2 = image_processing.resize_img(img2, 2)

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

    #hue_comparison = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    _, _, _, peak1_hist1 = cv2.minMaxLoc(hist1)
    _, _, _, peak1_hist2 = cv2.minMaxLoc(hist2)
    hist1[peak1_hist1[1],0] = 0
    hist2[peak1_hist2[1],0] = 0

    _, _, _, peak2_hist1 = cv2.minMaxLoc(hist1)
    _, _, _, peak2_hist2 = cv2.minMaxLoc(hist2)
    hist1[peak2_hist1[1], 0] = 0
    hist2[peak2_hist2[1], 0] = 0

    _, _, _, peak3_hist2 = cv2.minMaxLoc(hist2)
    hist2[peak3_hist2[1], 0] = 0

    peaks1 = [peak1_hist1[1], peak2_hist1[1]]
    peaks2 = [peak1_hist2[1], peak2_hist2[1], peak3_hist2[1]]

    common_peaks = 0
    for peak in peaks1:
        if np.isin(peak, peaks2):
            common_peaks = common_peaks + 1

    print('Common peaks: ', common_peaks)

    return common_peaks >= 2
