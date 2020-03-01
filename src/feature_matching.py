import cv2


def find_matches(des1, des2):

    ########################## FLANN ##########################
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
