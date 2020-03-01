import cv2
import image_processing as image_processing


def display_img(img, width=0, title='Image'):
    if width != 0:
        img = image_processing.resize_img_width(img, width)

    cv2.imshow(title, img)
    while True:
        key = cv2.waitKey(10)

        # Wait for ESC key
        if key == 27:
            break

    cv2.destroyAllWindows()


def draw_polygons(image, polygons):
    img_copy = image.copy()
    for polygon in polygons:
        polygon = polygon.reshape(polygon.shape[0], 2)
        cv2.polylines(img_copy, [polygon], True, (0, 255, 0), 5)

    return img_copy
