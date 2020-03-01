import cv2
import visualization


def find_matches(k1, kp2, des1, des2):

    ########################## FLANN ##########################
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)


    ########################## SIFT ##########################
    # bf = cv2.BFMatcher()
    # Find best k matches
    # start = time.time()
    # matches = bf.knnMatch(des1, des2, k=2)
    # t = time.time() - start
    # print('\t\t\t\tTIME MATCHING: ', t)


    # Draw matches
    matchesMask = [[0, 0] for i in range(len(matches))]

    for i, (match1, match2) in enumerate(matches):
        if match1.distance < 0.7 * match2.distance:
            matchesMask[i] = [1, 0]

    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), matchesMask=matchesMask, flags=0)
    flann_matches = cv2.drawMatchesKnn(source, kp1, scene, kp2, matches, None, **draw_params)
    visualization.display_img(flann_matches, width=800, title='Matches')


    # Keep only good matches
    good_matches = []
    for match1, match2 in matches:
        if match1.distance < 0.7 * match2.distance:
            good_matches.append(match1)


    return good_matches
