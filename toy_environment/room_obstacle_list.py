import numpy as np
from toy_environment. rectangle_object import RectangleObstacle

pre_obstacle_list = [
    (np.array([-0.5,  1.6]), np.array([ 0.05,  0.4 ])),
    (np.array([-0.5, -0.1]), np.array([ 0.05,  0.4 ])),
    #(np.array([-0.5 ,  0.75]), np.array([ 0.05,  0.45])),
    (np.array([-2.,  0.]), np.array([ 0.05,  2.1 ])),
    (np.array([ 2.,  0.]), np.array([ 0.05,  2.1 ])),
    (np.array([ 0.,  2.]), np.array([ 2.1 ,  0.05])),
    (np.array([ 0., -2.]), np.array([ 2.1 ,  0.05])),
    (np.array([ 2.,  2.]), np.array([ 0.1,  0.2])),
    (np.array([-2.,  2.]), np.array([ 0.2,  0.1])),
    (np.array([ 2., -2.]), np.array([ 0.2,  0.1])),
    (np.array([-2., -2.]), np.array([ 0.1,  0.2])),
    (np.array([ 0.75,  1.6 ]), np.array([ 1.25,  0.4 ])),
    (np.array([ 0.4,  0.2]), np.array([ 0.03,  0.03])),
    (np.array([-0.4,  0.2]), np.array([ 0.03,  0.03])),
    (np.array([ 0.4, -0.2]), np.array([ 0.03,  0.03])),
    (np.array([-0.4, -0.2]), np.array([ 0.03,  0.03])),
    (np.array([ 0.,  0.]), np.array([ 0.5,  0.3])),
    (np.array([ 1.4,  1.5]), np.array([ 0.05,  0.05])),
    (np.array([ 0.00724405,  0.00047065]),
     np.array([ 0.12458548,  0.22737432])),
    (np.array([-0.07262482,  0.00042312]),
     np.array([ 0.14067157,  0.18287495])),
    (np.array([ 0.01973567, -0.01924405]),
     np.array([ 0.03065044,  0.06756446])),
    (np.array([-0.03566579,  0.00042771]),
     np.array([ 0.23748511,  0.22563542])),
    (np.array([ 0.,  0.]),
     np.array([ 0.04 ,  0.015])),
    (np.array([ 0.,  0.]),
     np.array([ 0.04 ,  0.015])),
    (np.array([ 0.,  0.]),
     np.array([ 0.04 ,  0.015])),
    (np.array([ 0.,  0.]),
     np.array([ 0.04 ,  0.015])),
    (np.array([ 0.,  0.]),
     np.array([ 0.03,  0.  ])),
    (np.array([ 0.,  0.]), np.array([ 0.03,  0.  ])),
    (np.array([ 0.,  0.]), np.array([ 0.03,  0.  ])),
    (np.array([ 0.,  0.]), np.array([ 0.03,  0.  ])),
    (np.array([  2.99617112e-10,  -1.40730885e-10]),
     np.array([ 0.0202575,  0.0202575])),
    (np.array([  2.99617112e-10,  -1.40730885e-10]),
     np.array([ 0.0202575,  0.0202575])),
    (np.array([-0.039093  , -0.00035366]),
     np.array([ 0.07135277,  0.08104951])),
    (np.array([-0.03874661, -0.00035598]),
     np.array([ 0.06920656,  0.07723567])),
    (np.array([  2.21389409e-03,  -3.13193576e-05]),
     np.array([ 0.04224088,  0.06793738])),
    (np.array([  2.21389409e-03,  -3.13193576e-05]),
     np.array([ 0.04224088,  0.06793738])),
    (np.array([  5.49594413e-03,   1.64354635e-05]),
     np.array([ 0.02540056,  0.07078958])),
    (np.array([ -3.28302671e-02,   1.83681429e-05]),
     np.array([ 0.06936514,  0.08405513])),
    (np.array([-0.09445665, -0.00024871]),
     np.array([ 0.02890534,  0.09901419])),
    (np.array([ -5.54383682e-02,   3.47860699e-05]),
     np.array([ 0.07262519,  0.10925511])),
    (np.array([-0.07627559, -0.00039322]),
     np.array([ 0.0043356 ,  0.05954185])),
    (np.array([  1.18795196e-04,  -2.67583751e-05]),
     np.array([ 0.02127834,  0.02137586])),
    (np.array([  1.18795196e-04,  -2.67583751e-05]),
     np.array([ 0.02127834,  0.02137586])),
    (np.array([  2.20022990e-02,  -1.07730592e-08]),
     np.array([ 0.0124997 ,  0.01824984]))]

def process_obstacle_small_house(pos, size):
    pos = (pos + 2) / 4.
    size = size / 4.
    tl = [pos[0]-size[0], pos[1]-size[1]]
    br = [pos[0]+size[0], pos[1]+size[1]]
    return tl, br

def obstacle_list(image_size):
    obs_list = []
    for pos, size in pre_obstacle_list:
        tl, br = process_obstacle_small_house(pos, size)
        obs_list.append(RectangleObstacle(image_size, tl, br))
    return obs_list

