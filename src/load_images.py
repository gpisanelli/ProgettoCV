import cv2
import os.path

boxes_dict = {
    '0.jpg': 'Nesquik Cioccomilk',
    '1.jpg': 'Choco Krave (Blu)',
    '2.jpg': 'Corn Flakes (Giallo)',
    '3.jpg': 'Choko Goal',
    '4.jpg': 'Lim Chocolate',
    '5.jpg': 'Nesquik (Plain, Giallo)',
    '6.jpg': 'Nesquik Duo',
    '7.jpg': 'Coco Pops RisoCiok (Marrone)',
    '8.jpg': 'Coco Pops Palline (Rosso)',
    '9.jpg': 'Special K Classic (Bianco)',
    '10.jpg': 'Special K Fondente (Marrone)',
    '11.jpg': 'Choco Krave (Arancione)',
    '12.jpg': 'Fitness Fruits (Arancione)',
    '13.jpg': 'Fitness (Blu)',
    '14.jpg': 'Chocapic (Marrone)',
    '15.jpg': 'Coco Pops Rotelle (Giallo)',
    '16.jpg': 'Miel Pops Nocciola (Giallo)',
    '17.jpg': 'Miel Pops Anellini (Giallo)',
    '18.jpg': 'Country Crisp Chocolate (Marrone)',
    '19.jpg': 'Country Crisp Nuts (Blu)',
    '20.jpg': 'Special K Red Fruit (Rosso)',
    '21.jpg': 'Rice Krispies (Blu)',
    '22.jpg': 'Cheerios Miele (Viola)',
    '23.jpg': 'Special K Classic (Bianco)',
    '24.jpg': 'Fitness (Rosa)',
    '25.jpg': 'Coco Pops Palline (Rosso)',
    '26.jpg': 'Nesquik Duo (Rosa)'
}


def get_box_name(index):
    return boxes_dict[index]


def get_path_for_box(box_name):
    return os.path.join('..', 'images', 'object_detection_project', 'models', box_name)


def get_path_for_scene(scene_name):
    return os.path.join('..', 'images', 'object_detection_project', 'scenes', scene_name)


def load_img_color(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)


def load_img_grayscale(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def save_img(img, path):
    cv2.imwrite(path, img)


def load_all_imgs(dir_path):
    loaded_imgs = []
    with os.scandir(dir_path) as files:
        for filename in files:
            if filename.name.endswith('..jpg') and filename.is_file():
                loaded_imgs.append(load_img_color(filename.path))
    return loaded_imgs
