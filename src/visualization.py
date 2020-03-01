import cv2
import src.image_processing as image_processing


def display_img(img, width=0, title='Image'):
    if width != 0:
        img = image_processing.resize_img(img, width)

    cv2.imshow(title, img)
    while True:
        key = cv2.waitKey(10)
        # Wait for ESC key
        if key == 27:
            break

    cv2.destroyAllWindows()


def draw_polygons(image, polygons):
    for polygon in polygons:
        cv2.polylines(image, [polygon], True, (0, 255, 0), 10)
