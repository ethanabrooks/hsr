import numpy as np
from toy_environment.rectangle_object import RectangleObstacle
wall_width = 0.05

# (top_left, bottom_right)
pre_obstacle_list = [
    ([0.5-wall_width, 0], [0.5+wall_width, 2./6]),
    ([0.5-wall_width, 2./6], [0.5+wall_width, 4./6]),
    ([0.5-wall_width, 5./6], [0.5+wall_width, 7./6]),
    ([0, 0.5-wall_width], [1./6, 0.5+wall_width]),
    ([2./6, 0.5-wall_width], [4./6, 0.5+wall_width]),
    ([5./6, 0.5-wall_width], [7./6, 0.5+wall_width])
]

def obstacle_list(image_size):
    obstacles = []
    for (top_left, bottom_right) in pre_obstacle_list:
        obstacles.append(RectangleObstacle(image_size, top_left, bottom_right))
    return obstacles


