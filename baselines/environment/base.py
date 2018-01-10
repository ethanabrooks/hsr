"""Create gym environment for HSR"""

import os

import mujoco
import numpy as np
import time
from gym import spaces
from gym import utils

from environment.history_buffer import HistoryBuffer
from environment.server import Server


class BaseEnv(utils.EzPickle, Server):
    """ The environment """

    def __init__(self, geofence, max_steps,
                 xml_filepath, history_len, tb_dir, image_dimensions, use_camera, neg_reward,
                 steps_per_action, body_name, action_space=None, observation_space=None,
                 frames_per_step=20):
        utils.EzPickle.__init__(self)

        self._history_len = history_len
        self._history_buffer = HistoryBuffer(history_len)
        self._geofence = geofence
        self._body_name = body_name
        self._steps_per_action = steps_per_action
        self._frames_per_step = frames_per_step
        self._max_steps = max_steps
        self._use_camera = use_camera
        self._step_num = 0
        self._tb_dir = tb_dir
        self._neg_reward = neg_reward
        self._image_dimensions = image_dimensions
        self._goal = None

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

        if action_space is None:
            bounds = self.sim.actuator_ctrlrange.copy().reshape(-1, 2)
            low = bounds[:, 0]
            high = bounds[:, 1]
            self.action_space = spaces.Box(low, high)
        else:
            self.action_space = action_space

        if observation_space is None:
            observation, _reward, done, _info = self.step(np.zeros(self.sim.nu))
            assert not done
            self.obs_dim = observation.size

            high = np.inf * np.ones(self.obs_dim)
            low = -high
            self.observation_space = spaces.Box(low, high)
        else:
            self.observation_space = observation_space

        self._history_buffer.update(*self._obs())

    def server_values(self):
        return self.sim.qpos, self.sim.qvel

    def render(self, camera_name=None, labels=None):
        self.sim.render(camera_name, labels)

    def image(self, camera_name='rgb'):
        return self.sim.render_offscreen(
            *self._image_dimensions, camera_name)

    def obs(self):
        return self._vectorize_obs(self._history_buffer.get())

    def goal(self):
        return self._vectorize_goal(*self._goal)

    def step(self, action):
        self._step_num += 1
        step = 0
        reward = 0
        done = False

        while not done and step < self._steps_per_action:
            new_reward, done = self._step_inner(action)
            reward += new_reward
            step += 1

        self._history_buffer.update(*self._obs())
        return np.concatenate([self.obs(), self.goal()], axis=0), reward, done, {}
        #return (self.obs(), self.goal()), reward, done, {}

    def _step_inner(self, action):
        self.sim.ctrl[:] = action
        for _ in range(self._frames_per_step):
            self.sim.step()

        hit_max_steps = self._step_num >= self._max_steps
        done = False
        if self._terminal():
            #print('terminal')
            done = True
        elif hit_max_steps:
            #print('hit max steps')
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

        self._history_buffer.reset()
        self._history_buffer.update(*self._obs())
        #print(self.obs().shape, self.goal().shape)
        return np.concatenate([self.obs(), self.goal()], axis=0)
        #return self.obs(), self.goal()

    def _vectorize_obs(self, obs_history):
        """
        :param obs_history: values corresponding to output of self._obs_history
        :return: tuple of (values for cnn, values for mlp, goal)
        """
        mlp_history = [x for x in obs_history if len(x.shape) <= 1]
        mlp_array = np.concatenate(mlp_history, axis=-1).flatten()
        if self._use_camera:
            cnn_history = [x for x in obs_history if len(x.shape) > 1]
            cnn_array = np.concatenate(cnn_history, axis=-1)
            return mlp_array, cnn_array
        else:
            return mlp_array

    def _destructure_obs(self, mlp_input=None, cnn_input=None):
        shapes = self._history_buffer.shapes
        if cnn_input is not None:
            raise NotImplemented
        if mlp_input is not None:
            assert isinstance(mlp_input, np.ndarray), mlp_input
            mlp_shapes = [1 if len(shape) == 0 else shape[0]
                          for shape in shapes
                          if len(shape) <= 1]
            mlp_shapes_over_history = np.repeat(mlp_shapes, self._history_len)
            assert mlp_input.size == sum(mlp_shapes_over_history), \
                'mlp_input should not include `goal`.'
            indices = np.cumsum(mlp_shapes_over_history)
            raw_elements = np.split(mlp_input, indices)[:-1]
            by_type = np.split(np.array(raw_elements), len(mlp_shapes))
            observations = list(zip(*by_type))
            return observations[-1]
        raise RuntimeError("either data or images must not be None")

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

    def _vectorize_goal(self, *goal):
        raise NotImplemented

    def _destructure_goal(self, goal):
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
