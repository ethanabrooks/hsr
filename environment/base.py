"""Create gym environment for HSR"""

import os

import gym
import mujoco
import numpy as np
from gym import utils, spaces

from environment.history_buffer import HistoryBuffer
from environment.server import Server
from collections import deque


class BaseEnv(utils.EzPickle, Server):
    """ The environment """

    def __init__(self, geofence, max_steps,
                 xml_filepath, history_len, image_dimensions, use_camera, neg_reward,
                 steps_per_action, body_name,
                 frames_per_step=20):
        utils.EzPickle.__init__(self)

        self._history_buffer = deque(maxlen=history_len)
        self._geofence = geofence
        self._body_name = body_name
        self._steps_per_action = steps_per_action
        self._frames_per_step = frames_per_step
        self._max_steps = max_steps
        self._use_camera = use_camera
        self._step_num = 0
        self._neg_reward = neg_reward
        self._image_dimensions = image_dimensions

        # required for gym
        self.metadata = {}
        self.reward_range = -np.inf, np.inf
        self.spec = None

        fullpath = os.path.join(os.path.dirname(__file__), xml_filepath)
        if not fullpath.startswith("/"):
            fullpath = os.path.join(os.path.dirname(__file__),
                                    "assets", fullpath)
        self.sim = mujoco.Sim(fullpath)
        self.init_qpos = self.sim.qpos.ravel().copy()
        self.init_qvel = self.sim.qvel.ravel().copy()
        self._history_buffer += [self._obs()]
        self.observation_space = self.action_space = None

    def server_values(self):
        return self.sim.qpos, self.sim.qvel

    def render(self, camera_name=None, labels=None):
        self.sim.render(camera_name, labels)

    def image(self, camera_name='rgb'):
        return self.sim.render_offscreen(
            *self._image_dimensions, camera_name)

    def mlp_input(self):
        assert len(self._history_buffer) > 0
        obs_history = [np.concatenate(x, axis=0) for x in self._history_buffer]
        return np.concatenate(self._goal() + obs_history, axis=0)

    def destructure_mlp_input(self, mlp_input):
        assert isinstance(self.observation_space, gym.Space)
        assert self.observation_space.contains(mlp_input)
        goal_shapes = [np.size(x) for x in self._goal()]
        goal_size = sum(goal_shapes)

        # split mlp_input into goal and obs pieces
        goal_vector, obs_history = mlp_input[:goal_size], mlp_input[goal_size:]

        history_len = len(self._history_buffer)
        assert np.size(goal_vector) == goal_size
        assert (np.size(obs_history)) % history_len == 0

        # break goal vector into individual goals
        goals = np.split(goal_vector, goal_shapes, axis=0)

        # break history into individual observations in history
        history = np.split(obs_history, history_len, axis=0)

        obs_shapes = [np.size(x) for x in self._obs()]
        obs = []

        # break each observation in history into observation pieces
        for o in history:
            assert np.size(o) == sum(obs_shapes)
            obs += [np.split(o, obs_shapes, axis=0)]

        return goals, obs

    def step(self, action):
        assert np.shape(action) == np.shape(self.sim.ctrl)
        self._step_num += 1
        step = 0
        reward = 0
        done = False

        while not done and step < self._steps_per_action:
            new_reward, done = self._step_inner(action)
            reward += new_reward
            step += 1
        self._history_buffer.append(self._obs())
        return self.mlp_input(), reward, done, {}

    def _step_inner(self, action):
        assert np.shape(action) == np.shape(self.sim.ctrl)
        self.sim.ctrl[:] = action
        for _ in range(self._frames_per_step):
            self.sim.step()

        hit_max_steps = self._step_num >= self._max_steps
        done = False
        if self._terminal():
            # print('terminal')
            done = True
        elif hit_max_steps:
            # print('hit max steps')
            done = True
        elif self._currently_failed():
            done = True
        return self._current_reward(), done

    def reset(self):
        self.sim.reset()
        self._step_num = 0

        self._set_new_goal()
        qpos = self.reset_qpos()
        qvel = self.init_qvel + \
               np.random.uniform(size=self.sim.nv, low=-0.01, high=0.01)
        assert qpos.shape == (self.sim.nq,) and qvel.shape == (self.sim.nv,)
        self.sim.qpos[:] = qpos
        self.sim.qvel[:] = qvel
        self.sim.forward()
        return self.mlp_input()

    @staticmethod
    def seed(seed):
        np.random.seed(seed)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.sim.__exit__()

    def normalize(self, pos):
        raise RuntimeError("This doesn't work")

    def compute_reward(self, goal, args):
        raise NotImplemented

    def reset_qpos(self):
        raise NotImplemented

    def _current_reward(self):
        """Probably should call `compute_reward`"""
        raise NotImplemented

    def _set_new_goal(self):
        raise NotImplemented

    def _obs(self):
        raise NotImplemented

    def _goal(self):
        raise NotImplemented

    def _currently_failed(self):
        raise NotImplemented

    def _terminal(self):
        raise NotImplemented


def quaternion2euler(w, x, y, z):
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    euler_x = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = 1 if t2 > 1 else t2
    t2 = -1 if t2 < -1 else t2
    euler_y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    euler_z = np.arctan2(t3, t4)

    return euler_x, euler_y, euler_z


def distance_between(pos1, pos2):
    return np.sqrt(np.sum(np.square(pos1 - pos2)))


def at_goal(pos, goal, geofence):
    distance_to_goal = distance_between(pos, goal)
    return distance_to_goal < geofence


def escaped(pos, world_upper_bound, world_lower_bound):
    # noinspection PyTypeChecker
    return np.any(pos > world_upper_bound) \
           or np.any(pos < world_lower_bound)


def get_limits(pos, size):
    return pos + size, pos - size


def point_inside_object(point, object):
    pos, size = object
    tl = pos - size
    br = pos + size
    return (tl[0] <= point[0] <= br[0]) and (tl[1] <= point[1] <= br[1])


def print1(*strings):
    print('\r', *strings, end='')
