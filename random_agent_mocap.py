#! /usr/bin/env python3
"""Agent that executes random actions"""

import argparse
import numpy as np
# import sys
# import scipy.misc
# import matplotlib.pyplot as plt
# import os
import time

from environment.base import print1
from environment.navigate import NavigateEnv
from environment.pick_and_place_mocap import PickAndPlaceMocapEnv
from environment.server import Server


def run(port):
    image_dimensions = 32, 32, 3
    # env = NavigateEnv(
    #     continuous_actions=True,
    #     steps_per_action=100,
    #     geofence=.3,
    #     image_dimensions=image_dimensions[:2])
    with PickAndPlaceMocapEnv(max_steps=999999,
                         use_camera=False,
                         geofence=.000003,
                         image_dimensions=image_dimensions[:2]) as env:
        if port:
            server = Server(port)
        env.reset()
        shape = env.action_space.shape
        shape, = shape
        print(shape)
        action = np.zeros(shape,)
        delta = .02
        i = 0
        # image = np.zeros(image_dimensions)
        # im = plt.imshow(image)
        action = np.ones(5)
        action = action * np.random.rand(1)
        action += delta
        if np.any(action > 1) or np.any(action < -1):
            delta *= -1
        action[0:4] = 0
        while True:
            if i != 0:
                env.render()

            action[4] = -.5
            action[1] = 120
            
            i += 1

            action[3] = 1.46
            if i > 200:
                action[3] = -1.46
                action[1] = -120
                action[4] = -.25
            if i > 300:
                action[3] = 1.05
                action[1] = 120
                action[4] = -.5
            if i > 400:
                action[3] = -.12
                action[1] = -120
                action[4] = -.5

            tick = time.time()
            print(action, i)
            obs, r, done, _ = env.step(action)

            # NOTE: this is how to matplotlib render inner cameras. Do not delete.
            # action += 1e-2
            # if np.any(action >= 1):
            # print('boop')
            #     action = -np.ones(shape)
            # if done:
            # env.reset()
            # j += 1
            # image = env.image
            # im.set_data(image)
            # plt.pause(0.05)
            # plt.draw()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', type=int, default=None, help='port')
    args = parser.parse_args()

    run(args.port)
