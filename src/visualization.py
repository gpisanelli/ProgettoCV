import cv2


def display_img(img, maxSize=0, title='Image'):
    if maxSize != 0:
        if img.shape[0] * 16 / 9 <= img.shape[1]:
            width = 1000
            height = int(width * img.shape[0] / img.shape[1])
            dim = (width, height)
            img = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
        else:
            height = 700
            width = int(height * img.shape[1] / img.shape[0])
            dim = (width, height)
            img = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)

    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
