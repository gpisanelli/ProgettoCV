import sys

from utils import load_images, visualization
import pipeline_easy
import pipeline_hard
import pipeline_medium

scene_dict = {
    '-e': ['e1.png', 'e2.png', 'e3.png', 'e4.png', 'e5.png'],
    '-m': ['m1.png', 'm2.png', 'm3.png', 'm4.png', 'm5.png'],
    '-h': ['h1.jpg', 'h2.jpg', 'h3.jpg', 'h4.jpg', 'h5.jpg']
}

box_names_dict = {
    '-e': ['0.jpg', '1.jpg', '11.jpg', '19.jpg', '24.jpg', '25.jpg', '26.jpg'],
    '-m': ['0.jpg', '1.jpg', '11.jpg', '19.jpg', '24.jpg', '25.jpg', '26.jpg'],
    '-h': ['0.jpg', '1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6.jpg', '7.jpg',
           '8.jpg', '9.jpg', '10.jpg', '11.jpg', '12.jpg', '13.jpg', '14.jpg', '15.jpg',
           '16.jpg', '17.jpg', '18.jpg', '19.jpg', '20.jpg', '21.jpg', '22.jpg', '23.jpg']
}


def main():
    accepted_list = ['-e', '-m', '-h']
    arg = sys.argv[1]
    if len(sys.argv) != 2 or arg not in accepted_list:
        print('usage: ./main.py -(e|h|m)')
        sys.exit(2)

    scene_names = scene_dict[arg]
    scenes = []

    for scene_name in scene_names:
        scene_path = load_images.get_path_for_scene(scene_name)
        scenes.append(load_images.load_img_color(scene_path))

    if arg == '-e':
        pipeline_easy.start(box_names_dict['-e'], scenes)

    elif arg == '-m':
        pipeline_medium.start(box_names_dict['-m'], scenes)

    elif arg == '-h':
        pipeline_hard.start(box_names_dict['-h'], scenes)


main()
