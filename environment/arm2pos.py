import numpy as np

from environment.base import at_goal
from environment.pick_and_place import PickAndPlaceEnv


def achieved_goal(goal_pos, gripper_pos, geofence):
    return at_goal(gripper_pos, goal_pos, geofence)


class Arm2Pos(PickAndPlaceEnv):
    def __init__(self, *args, **kwargs):
        PickAndPlaceEnv.__init__(self, *args, **kwargs)
        self._set_new_goal()

    @staticmethod
    def achieved_goal(goal_pos, should_grasp, gripper_pos, block_lifted, geofence):
        return at_goal(gripper_pos, goal_pos, geofence)

    def _set_new_goal(self):
        low = np.array([-.273, -.552, .506])
        high = np.array([.117, -.052, .506])
        self._goal = np.random.random(size=3) * (high - low) / 2. + (high + low) / 2.
        assert np.all(low <= self.__goal) and np.all(self.__goal <= high)

    @property
    def _goal(self):
        return self.__goal, True

    @_goal.setter
    def _goal(self, value):
        self.__goal = value

