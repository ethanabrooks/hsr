import os

import mujoco
import numpy as np

from environment.base import BaseEnv


class MujocoEnv(BaseEnv):
    def __init__(self, xml_filepath, max_steps, history_len, image_dimensions,
                 neg_reward, steps_per_action, frames_per_step=20):
        fullpath = os.path.join(os.path.dirname(__file__), xml_filepath)
        if not fullpath.startswith("/"):
            fullpath = os.path.join(os.path.dirname(__file__), "assets", fullpath)
        self.sim = mujoco.Sim(fullpath)
        self.init_qpos = self.sim.qpos.ravel().copy()
        self.init_qvel = self.sim.qvel.ravel().copy()
        self._frames_per_step = frames_per_step
        super().__init__(max_steps, history_len, image_dimensions,
                         neg_reward, steps_per_action)