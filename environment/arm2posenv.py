import numpy as np
from gym import spaces
from os.path import join

from environment.base import at_goal, BaseEnv
from environment.pick_and_place import PickAndPlaceEnv


def achieved_goal(goal_pos, gripper_pos, geofence):
    return at_goal(gripper_pos, goal_pos, geofence)


class Arm2PosEnv(BaseEnv):
    def __init__(self, continuous, max_steps, geofence=.08, history_len=1, neg_reward=True,
                 image_dimensions=None, action_multiplier=1):

        BaseEnv.__init__(self,
                         geofence=geofence,
                         max_steps=max_steps,
                         xml_filepath=join('models', 'pick-and-place', 'world.xml'),
                         history_len=history_len,
                         use_camera=False,  # TODO
                         neg_reward=neg_reward,
                         body_name="hand_palm_link",
                         steps_per_action=10,
                         image_dimensions=image_dimensions)

        self._action_multiplier = action_multiplier
        self._continuous = continuous

        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=self._obs()[0].shape[0] + 3)

        if continuous:
            self.action_space = spaces.Box(-1, 1, shape=self.sim.nu)
        else:
            self.action_space = spaces.Discrete(self.sim.nu * 2 + 1)
        self.__goal = [self._new_goal()]
        left_finger_name = 'hand_l_distal_link'
        self._finger_names = [left_finger_name, left_finger_name.replace('_l_', '_r_')]

    def _gripper_pos(self, qpos=None):
        finger1, finger2 = [self.sim.get_body_xpos(name, qpos)
                            for name in self._finger_names]
        return (finger1 + finger2) / 2.

    def _current_reward(self):
        return self._compute_reward(self._goal(), self._gripper_pos())

    def _compute_reward(self, goal_pos, gripper_pos):
        if at_goal(gripper_pos, goal_pos, self._geofence):
            return 1
        elif self._neg_reward:
            return -.0001
        else:
            return 0

    def _set_new_goal(self):
        self.__goal = [self._new_goal()]

    def _new_goal(self):
        # [-0.02368331  0.31957946  0.5147059]
        # [-0.02229058 - 0.17246746  0.50834088]
        high = np.array([-.022, .32, .51])
        low = np.array([-.022, -.18, .51])
        goal = np.random.uniform(low, high)
        assert np.all(low <= goal) and np.all(goal <= high)
        return goal

    def _obs(self):
        return [self.sim.qpos]

    def _goal(self):
        return self.__goal

    def _currently_failed(self):
        return False

    def _terminal(self):
        return at_goal(self._gripper_pos(), self._goal(), self._geofence)

    def step(self, action):
        if not self._continuous:
            ctrl = np.zeros(self.sim.nu)
            if action != 0:
                ctrl[(action - 1) // 2] = 1 if action % 2 else -1
            return BaseEnv.step(self, ctrl)
        else:
            action = np.clip(action * self._action_multiplier, -1, 1)
            return BaseEnv.step(self, action)

    def reset_qpos(self):
        return self.init_qpos
