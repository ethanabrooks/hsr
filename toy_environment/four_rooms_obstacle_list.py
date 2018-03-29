import numpy as np
from toy_environment.rectangle_object import RectangleObstacle
wall_width = 0.05
# (top_left, bottom_right)
train_obstacle_list = [
    ([0.5-wall_width, -1], [0.5+wall_width, 2./6]),
    ([0.5-wall_width, 2./6], [0.5+wall_width, 4./6]),
    ([0.5-wall_width, 4./6], [0.5+wall_width, 10./6]),
    ([-1, 0.5-wall_width], [2./6, 0.5+wall_width]),
    ([2./6, 0.5-wall_width], [4./6, 0.5+wall_width]),
    ([4./6, 0.5-wall_width], [10./6, 0.5+wall_width]) 
]

eval_obstacle_list = [
    ([0.5-wall_width, -1], [0.5+wall_width, 2./6]),
    ([0.5-wall_width, 2./6], [0.5+wall_width, 4./6]),
    ([0.5-wall_width, 5./6], [0.5+wall_width, 10./6]),
    ([-1, 0.5-wall_width], [1./6, 0.5+wall_width]),
    ([2./6, 0.5-wall_width], [4./6, 0.5+wall_width]),
    ([5./6, 0.5-wall_width], [10./6, 0.5+wall_width])
]

def obstacle_list(image_size, eval_=False):
    obstacles = []
    if not eval_:
        pre_obstacle_list = train_obstacle_list
    else:
        pre_obstacle_list = eval_obstacle_list

    for (top_left, bottom_right) in pre_obstacle_list:
        obstacles.append(RectangleObstacle(image_size, top_left, bottom_right))
    return obstacles


