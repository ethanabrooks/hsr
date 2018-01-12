import numpy as np
from gym import spaces
from os.path import join

from environment.base import at_goal, BaseEnv
from environment.pick_and_place import PickAndPlaceEnv


def achieved_goal(goal_pos, gripper_pos, geofence):
    return at_goal(gripper_pos, goal_pos, geofence)


class Arm2Pos(PickAndPlaceEnv):
    def __init__(self, continuous, *args, geofence=.08, **kwargs):
        PickAndPlaceEnv.__init__(self, *args, geofence=geofence, **kwargs)
        self._continuous = continuous
        if not continuous:
            self.action_space = spaces.Discrete(self.sim.nu * 2 + 1)
        self._set_new_goal()

    @staticmethod
    def achieved_goal(goal_pos, should_grasp, gripper_pos, block_lifted, geofence):
        return at_goal(gripper_pos, goal_pos, geofence)

    def _set_new_goal(self):
        # [-0.02368331  0.31957946  0.5147059]
        # [-0.02229058 - 0.17246746  0.50834088]
        high = np.array([-.022, .32, .51])
        low = np.array([-.022, -.18, .51])
        self._goal = np.random.uniform(low, high)
        assert np.all(low <= self.__goal) and np.all(self.__goal <= high)

    @property
    def _goal(self):
        return self.__goal, True

    @_goal.setter
    def _goal(self, value):
        self.__goal = value

    def step(self, action):
        if not self._continuous:
            ctrl = np.zeros(self.sim.nu)
            if action != 0:
                ctrl[(action - 1) // 2] = 1 if action % 2 else -1
            return BaseEnv.step(self, ctrl)
        else:
            return PickAndPlaceEnv.step(self, action)
