from utils import visualization, load_images


def display_result(found_instances, visualization_scene, draw_names=True):

    for name in found_instances:
        if len(found_instances[name]) > 0:
            print(name, '-', load_images.get_box_name(name))
            print('Found instances: ', len(found_instances[name]))

            for rect in found_instances[name]:
                x, y, w, h = rect
                barycenter = ((x + w // 2), (y + h // 2))
                print('width:', w, '  height:', h, '  barycenter:', barycenter)

                visualization_scene = visualization.draw_bounding_rect(visualization_scene, rect)
                if draw_names:
                    visualization_scene = visualization.draw_names(visualization_scene, rect, name)

            print('\n')

    visualization.display_img(visualization_scene)
