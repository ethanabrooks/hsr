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
from environment.pick_and_place import PickAndPlaceEnv
from environment.server import Server


def run(port):
    image_dimensions = 32, 32, 3
    # env = NavigateEnv(
    #     continuous_actions=True,
    #     steps_per_action=100,
    #     geofence=.3,
    #     image_dimensions=image_dimensions[:2])
    with PickAndPlaceEnv(max_steps=999999,
                         use_mocap=False) as env:
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
        # action[0:4] = 0
        while True:
            if i != 0:
                env.render()

            action = np.zeros(5)
            action[0] = 45
            action[1] = 0.6
            action[2] = 0.1
            action[3] = 0
            action[4] = 0.0
            
            i += 1

            if i > 25: 
                action[0] = 0
                action[1] = -1
                action[2] = 0.1
                action[3] = 0.0
                action[4] = 0.0

            if i > 100:
                action[0] = -120
                action[1] = -.1
                action[2] = 0.1
                action[3] = 0.0
                action[4] = 0.0

            tick = time.time()
            # print(action, i)
            obs, r, done, _ = env.step(action)
            # print(obs)

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
