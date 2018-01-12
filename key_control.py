#! /usr/bin/env python3
"""Agent that executes random actions"""
# import gym
import argparse

import numpy as np
from mujoco import ObjType

from environment.arm2posenv import Arm2PosEnv

saved_pos = None


def run():
    env = Arm2PosEnv(continuous=False, max_steps=9999999, neg_reward=True, use_camera=False, action_multiplier=.01)

    while True:
        try:
            action = int(env.sim.get_last_key_press())
            assert env.action_space.contains(action)
        except (KeyError, TypeError, AssertionError):
            action = 0
        assert isinstance(action, int)
        obs, r, done, _ = env.step(action)
        env.render()

        if done:
            env.reset()
            print('\nresetting')

        assert not env._currently_failed()
        assert_equal(env._goal, env._destructure_goal(env._vector_goal()))
        assert_equal(env._obs(), env._destructure_obs(env._vector_obs()))
        assert_equal(env._gripper_pos(), env._gripper_pos(env.sim.qpos), atol=1e-2)


def assert_equal(val1, val2, atol=1e-5):
    for a, b in zip(val1, val2):
        assert np.allclose(a, b, atol=atol), "{} vs. {}".format(a, b)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', type=int, default=None)
    args = parser.parse_args()

    run()
