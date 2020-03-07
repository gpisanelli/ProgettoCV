import cv2

import image_processing
import load_images
import visualization


def main():
    hough = cv2.createGeneralizedHoughGuil()
    hough.setMaxAngle(5)
    hough.setMinAngle(0)
    hough.setAngleStep(1)
    hough.setMaxScale(4)
    hough.setMinScale(0.1)
    hough.setScaleStep(0.01)
    hough.setMinDist(10)
    hough.setLevels(100)
    hough.setAngleThresh(1000)
    hough.setScaleThresh(100)
    hough.setPosThresh(100)
    hough.setDp(2)
    hough.setMaxBufferSize(1000)

    template_path = '../images/generalized_hough_demo_02.png'
    template = load_images.load_img_grayscale(template_path)

    scene_path = '../images/generalized_hough_demo_01.png'
    scene = load_images.load_img_grayscale(scene_path)

    hough.setTemplate(template, (int(template.shape[1]/2), int(template.shape[0]/2)))
    positions, votes = hough.detect(scene)

    for position in positions:
        print(position)
        cv2.polylines()
        cv2.circle(scene, (int(position[0][0]),int(position[0][1])), 10, 100, 3)

    visualization.display_img(scene)


main()