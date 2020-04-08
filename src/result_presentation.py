from utils import visualization


def display_result(found_instances, visualization_scene):

    for name in found_instances:
        for rect in found_instances[name]:
            visualization_scene = visualization.draw_bounding_rect(visualization_scene, rect)
            # visualization_scene = visualization.draw_names(visualization_scene, bounds, name)

    visualization.display_img(visualization_scene, width=1000)