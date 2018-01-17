#! /usr/bin/env python3
"""Agent that executes random actions"""
# import gym
import argparse

import numpy as np
from mujoco import ObjType

from environment.arm2pos import Arm2PosEnv
from environment.base import distance_between

saved_pos = None


def run():
    env = Arm2PosEnv(continuous=False, max_steps=9999999, neg_reward=True, action_multiplier=.1)
    total_r = 0

    while True:
        keypress = env.sim.get_last_key_press()
        if keypress == ' ':
            # print(distance_between(env._gripper_pos(), env._goal[0]))
            print(env._gripper_pos())
        try:
            action = int(keypress)
            assert env.action_space.contains(action)
        except (KeyError, TypeError, AssertionError, ValueError):
            action = 0
        assert isinstance(action, int)
        obs, r, done, _ = env.step(action)
        total_r += r

        if done:
            print(total_r)
            total_r = 0
            env.reset()
            print('\nresetting')

        env.render(labels={'x': env._goal()[0]})
        assert not env._currently_failed()


def assert_equal(val1, val2, atol=1e-5):
    for a, b in zip(val1, val2):
        assert np.allclose(a, b, atol=atol), "{} vs. {}".format(a, b)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', type=int, default=None)
    args = parser.parse_args()

    run()
