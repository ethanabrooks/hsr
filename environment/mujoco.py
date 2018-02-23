import os

import mujoco

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
        BaseEnv.__init__(self, xml_filepath, max_steps, history_len, image_dimensions,
                         neg_reward, steps_per_action, frames_per_step)

    def server_values(self):
        return self.sim.qpos, self.sim.qvel

    def render(self, mode=None, camera_name=None, labels=None):
        if mode == 'rgb_array':
            return self.sim.render_offscreen(height=256, width=256)
        self.sim.render(camera_name, labels)

    def image(self, camera_name='rgb'):
        return self.sim.render_offscreen(
            *self._image_dimensions, camera_name)
