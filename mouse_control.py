#! /usr/bin/env python3
"""Agent that executes random actions"""
# import gym
import argparse

import numpy as np
from mujoco import ObjType

from environment.navigate import NavigateEnv
from environment.pick_and_place import PickAndPlaceEnv

saved_pos = None


def run(port, value_tensor=None, sess=None):
    env = NavigateEnv(continuous_actions=True, steps_per_action=100, geofence=.3,
                      use_camera=False, action_multiplier=.1)

    # env = Arm2PosEnv(action_multiplier=.01, history_len=1, continuous=True, max_steps=9999999, neg_reward=True)
    # env = PickAndPlaceEnv(max_steps=9999999, action_multiplier=.0001)
    env.reset()

    shape, = env.action_space.shape

    i = 0
    action = np.zeros(shape)
    moving = False
    total_reward = 0

    while True:
        lastkey = env.sim.get_last_key_press()
        if moving:
            action[i] += env.sim.get_mouse_dy()
        else:
            for name in ['wrist_roll_motor']:  # , 'slide_x_motor', 'slide_y_motor']:
                k = env.sim.name2id(ObjType.ACTUATOR, name)
                action[k] = 0
        if lastkey is ' ':
            moving = not moving
            print('\rmoving:', moving)
        if lastkey is 'P':
            print(env._gripper_pos())

        for k in range(10):
            if lastkey == str(k):
                i = k - 1
                print('')
                print(env.sim.id2name(ObjType.ACTUATOR, i))

        obs, r, done, _ = env.step(action)
        total_reward += r
        run_tests(env, obs)

        if done:
            env.reset()
            print('\nresetting', total_reward)
            total_reward = 0
        env.render(labels={'x': np.ravel(env._goal()[0])})


def run_tests(env, obs):
    assert env.observation_space.contains(obs)
    assert not env._currently_failed()
    assert np.shape(env._goal()) == np.shape(env.obs_to_goal(obs))
    goal, obs_history = env.destructure_mlp_input(obs)
    assert_equal(env._goal(), goal)
    assert_equal(env._obs(), obs_history[-1])
    assert_equal((goal, obs_history), env.destructure_mlp_input(env.mlp_input(goal, obs_history)))
    assert_equal(obs, env.mlp_input(*env.destructure_mlp_input(obs)))
    assert_equal(obs, env.change_goal(goal, obs))
    assert_equal(env._gripper_pos(), env._gripper_pos(env.sim.qpos), atol=1e-2)


def assert_equal(val1, val2, atol=1e-5):
    try:
        for a, b in zip(val1, val2):
            assert_equal(a, b, atol=atol)
    except TypeError:
        assert np.allclose(val1, val2, atol=atol), "{} vs. {}".format(val1, val2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', type=int, default=None)
    args = parser.parse_args()

    run(args.port)
