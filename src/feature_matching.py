import cv2
import numpy as np


def find_matches(des1, des2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Keep only good matches
    good_matches = []
    for match1, match2 in matches:
        if match1.distance < 0.7 * match2.distance:
            good_matches.append(match1)

    return good_matches


def find_object(matches, kp1, kp2, scene_img):
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h = scene_img.shape[0]
    w = scene_img.shape[1]

    # Find bounds of the object in the scene
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    bounds = np.int32(dst)
    bounds = bounds.reshape(bounds.shape[0], 2)

    return bounds, M



def validate_match():
    pass
