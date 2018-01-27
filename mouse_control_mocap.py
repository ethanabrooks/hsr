#! /usr/bin/env python3
"""Agent that executes random actions"""
# import gym
import argparse

import numpy as np
import tensorflow as tf
from mujoco import ObjType

from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.ddpg import ddpg
from baselines.ddpg.models import Critic
from environment.base import print1
from environment.pick_and_place_mocap import PickAndPlaceMocapEnv

saved_pos = None


def run(port, restore_critic_path=None):
    np.set_printoptions(precision=3, linewidth=1000)
    # env = NavigateEnv(continuous_actions=True, steps_per_action=100, geofence=.3,
    #                   use_camera=False, action_multiplier=.1, image_dimensions=image_dimensions[:2])

    env = PickAndPlaceMocapEnv(max_steps=9999999, neg_reward=True, use_camera=False, action_multiplier=.01)

    mocap_action_space_labels = [
        'gripper_dist',
        'desired_rotation',
        'mocap_x',
        'mocap_y',
        'mocap_z',
    ]

    obs, goal = env.reset()
    if port:
        env.start_server(port)
    shape, = env.action_space.shape
    i = 0
    action = np.zeros(shape)
    moving = False
    last = {}
    pause = False

    if restore_critic_path:
        obs_placeholder = tf.placeholder(tf.float32, obs.shape)
        goal_placeholder = tf.placeholder(tf.float32, goal.shape)
        act_placeholder = tf.placeholder(tf.float32, action.shape)

        layer_norm = tf.get_variable('layer_norm', shape=(), dtype=bool)
        normalize_observations = tf.get_variable('normalize_observations', shape=(), dtype=bool)
        use_cnn = tf.get_variable('use_cnn', shape=(), dtype=bool)
        name_critic = tf.get_variable('name_critic', shape=(),
                                      dtype=tf.string, initializer=tf.zeros_initializer())

        saver = tf.train.Saver([layer_norm, normalize_observations, use_cnn, name_critic])
        with tf.Session() as sess:
            saver.restore(sess, restore_critic_path)
            layer_norm, normalize_observations, use_cnn, critic_name = sess.run([
                layer_norm,
                normalize_observations,
                use_cnn,
                name_critic,
            ])

        if normalize_observations:
            shape = env.obs().shape
            with tf.variable_scope('obs_rms'):
                rms = RunningMeanStd(shape=shape)
        else:
            rms = None
        normalized_obs = ddpg.preprocess_obs(obs=tf.expand_dims(obs_placeholder, 0),
                                             observation_range=(-5., 5.), rms=rms)

        critic = Critic(name=critic_name.decode('utf-8'),
                        layer_norm=layer_norm,
                        use_cnn=use_cnn)

        value_tensor = critic(obs=normalized_obs,
                              goal=tf.expand_dims(goal_placeholder, 0),
                              action=tf.expand_dims(act_placeholder, 0))
        saver = tf.train.Saver()

    try:
        if restore_critic_path:
            sess = tf.Session()
            saver.restore(sess, restore_critic_path)
        while True:
            lastkey = env.sim.get_last_key_press()
            if moving:
                action[i] += env.sim.get_mouse_dy()
                print(action[i])
            else:
                for name in ['slide_x_motor', 'slide_y_motor', 'turn_motor']:
                    k = env.sim.name2id(ObjType.ACTUATOR, name)
                    action[k] = 0
            if lastkey is ' ':
                moving = not moving
                print('\rmoving:', moving)

            if lastkey is 'A':
                last['qpos'] = env.sim.qpos.copy()
                last['qvel'] = env.sim.qvel.copy()
                last['ctrl'] = env.sim.ctrl.copy()
                print('\nsaved:')
                print(last['qpos'])
                pause = True
            if lastkey is 'P':
                pause = not pause
            if lastkey is 'D':
                env.step(action)
            if lastkey is 'Z':
                action[:] = 0

            for k in range(10):
                if lastkey == str(k):
                    i = k - 1
                    print(mocap_action_space_labels[i])

            # action[1:4] = 0
            if not pause:
                # print(action)
                # new_action[2] += 0.001
                (obs, goal), r, done, _ = env.step(action)
                # print(obs)
                labels = dict()
                qpos, _ = env._destructure_obs(obs)
                if restore_critic_path:
                    value = sess.run(value_tensor,
                                     feed_dict={obs_placeholder: obs,
                                                goal_placeholder: goal,
                                                act_placeholder: action, })

                    labels = {value[0, 0]: env._gripper_pos(),
                              env._step_num: (0, 0, 0)}

                env.render(labels=labels)

                # if done:
                #     env.reset()
                #     print('\nresetting')

            assert not env._currently_failed()
            assert_equal(env._goal, env._destructure_goal(env.goal()))
            assert_equal(env._obs(), env._destructure_obs(env.obs()))
            assert_equal(env._gripper_pos(), env._gripper_pos(env.sim.qpos), rtol=1e-2)
    finally:
        if restore_critic_path:
            sess.close()


def assert_equal(val1, val2, rtol=1e-5):
    for a, b in zip(val1, val2):
        assert np.allclose(a, b, rtol=rtol),  "{} vs. {}".format(a, b)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', type=int, default=None)
    parser.add_argument('-r', '--restore-critic-path', type=str, default=None)
    args = parser.parse_args()

    run(args.port, args.restore_critic_path)
