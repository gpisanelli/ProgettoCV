import cv2
import numpy as np
from utils import visualization


def detect_features_SIFT(img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp_t, des_t = sift.detectAndCompute(img, None)
    return kp_t, des_t


def find_matches(des1, des2, dist_threshold=0.7):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Keep only good matches
    good_matches = []
    for match1, match2 in matches:
        if match1.distance < dist_threshold * match2.distance:
            good_matches.append(match1)

    return good_matches


def find_object(matches, kp1, kp2, box_img):
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Aggiungere controllo eccezioni
    M, matches_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
    if M is not None:
        h = box_img.shape[0]
        w = box_img.shape[1]
        # Find bounds of the object in the scene
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        bounds = np.int32(dst)
        bounds = bounds.reshape(bounds.shape[0], 2)

        matches_arr = np.asarray(matches).reshape((len(matches), 1))
        used_matches = matches_arr[matches_mask > 0]
        used_src_pts = np.float32([kp1[m.queryIdx].pt for m in used_matches])
        used_dst_pts = np.float32([kp2[m.trainIdx].pt for m in used_matches])

        not_used_matches = matches_arr[matches_mask == 0]
        not_used_matches = not_used_matches.tolist()

        return True, bounds, M, used_src_pts, used_dst_pts, not_used_matches
    else:
        return False, None, None, None, None, None



def find_object_similarity_functions(img, template):
    img2 = img.copy()
    w = template.shape[1]
    h = template.shape[0]

    # All the 6 methods for comparison in a list
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
               'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    for meth in methods:
        img = img2.copy()
        method = eval(meth)

        # Apply template Matching
        res = cv2.matchTemplate(cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))[0], cv2.split(cv2.cvtColor(template, cv2.COLOR_BGR2HSV))[0], method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv2.rectangle(img, top_left, bottom_right, 255, 2)

        visualization.display_img(img, title=meth)


def validate_match():
    pass