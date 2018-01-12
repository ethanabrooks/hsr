from os.path import join

import numpy as np
from gym import spaces
from mujoco import ObjType

from environment.base import BaseEnv, at_goal, print1


def failed(resting_block_height, goal_block_height):
    return False
    # return resting_block_height - goal_block_height > .02  #.029


class PickAndPlaceEnv(BaseEnv):
    def __init__(self, max_steps, use_camera, geofence=.05, tb_dir=None, neg_reward=True, history_len=4,
                 image_dimensions=(64, 64), action_multiplier=1):

        self._dimensions = 64, 64
        self._action_multiplier = action_multiplier
        self._history_len = history_len
        self._goal_block_name = 'block1'
        self._min_lift_height = 0.02
        self._resting_block_height = .49  # empirically determined
        left_finger_name = 'hand_l_distal_link'
        self._finger_names = [left_finger_name, left_finger_name.replace('_l_', '_r_')]

        super().__init__(
            geofence=geofence,
            max_steps=max_steps,
            xml_filepath=join('models', 'pick-and-place', 'world.xml'),
            history_len=history_len,
            tb_dir=tb_dir,
            use_camera=use_camera,
            neg_reward=neg_reward,
            body_name="hand_palm_link",
            steps_per_action=10,
            image_dimensions=image_dimensions)

        if use_camera:
            self.observation_space = spaces.Box(
                0, 1, shape=(list(self._dimensions) + [3 * history_len]))
        else:
            self.observation_space = spaces.Box(-np.inf, np.inf,
                                                shape=self.obs().shape[0] + 4)

        self.action_space = spaces.Box(-1, 1, shape=self.sim.nu - 1)
        self._table_height = self.sim.get_body_xpos('pan')[2]

    # goal stuff

    @property
    def _goal(self):
        return self.sim.get_body_xpos(self._goal_block_name), True

    @_goal.setter
    def _goal(self, value):
        pass

    def _set_new_goal(self):
        pass

    @staticmethod
    def achieved_goal(goal_pos, should_grasp, gripper_pos, block_lifted, geofence):
        return at_goal(gripper_pos, goal_pos, geofence) and should_grasp == block_lifted

    def _vectorize_goal(self, goal, should_grasp):
        return np.append(goal, should_grasp)

    def _destructure_goal(self, goal):
        return goal[:-1], goal[-1] == 1

    # obs stuff

    def _obs(self):
        block_lifted, should_grasp = map(np.array, [[self._block_lifted()], [True]])
        return self.sim.qpos, block_lifted

    def _obs_to_goal(self, qpos, block_lifted):
        """
        :return: goal that would make obs be `at_goal`
        """
        should_grasp = block_lifted  # reward whatever the agent actually did
        return self._gripper_pos(qpos), should_grasp

    def obs_to_goal(self, data=None, images=None):
        return self._vectorize_goal(*self._obs_to_goal(*self._destructure_obs(data, images)))

    # positions

    def _gripper_pos(self, qpos=None):
        finger1, finger2 = [self.sim.get_body_xpos(name, qpos)
                            for name in self._finger_names]
        return (finger1 + finger2) / 2.

    def _goal_block_height(self, qpos=None):
        x, y, z = self.sim.get_body_xpos(self._goal_block_name, qpos)
        return z

    def _block_lifted(self):
        return self._goal_block_height() - self._resting_block_height > self._min_lift_height

    # terminal stuff

    def _terminal(self):
        return self.achieved_goal(*self._goal, self._gripper_pos(),
                                  self._block_lifted(), self._geofence)

    def compute_terminal(self, goal, mlp_input=None, cnn_input=None):
        if cnn_input:
            raise NotImplemented
        goal_pos, should_grasp = self._destructure_goal(goal)
        qpos, block_lifted = self._destructure_obs(mlp_input)
        gripper_pos = self._gripper_pos(qpos)
        return self.achieved_goal(goal_pos, should_grasp, gripper_pos, block_lifted, self._geofence)

    def _stuck(self):
        return False

    def _currently_failed(self):
        return failed(self._resting_block_height, self._goal_block_height())

    def reset_qpos(self):
        return self.init_qpos

    # reward stuff

    def _current_reward(self):
        return self._compute_reward(*self._goal, self._gripper_pos(), self._block_lifted(), self._currently_failed())

    def _compute_reward(self, goal_pos, should_grasp, gripper_pos, block_lifted, failed):
        if at_goal(gripper_pos, goal_pos, self._geofence) and should_grasp == block_lifted:
            return 1
        elif failed:
            return -1
        elif self._neg_reward:
            return -.0001
        else:
            return 0

    def compute_reward(self, goal, mlp_input=None, cnn_input=None):
        if cnn_input:
            raise NotImplemented
        goal_pos, should_grasp = self._destructure_goal(goal)
        qpos, block_lifted = self._destructure_obs(mlp_input)
        gripper_pos = self._gripper_pos(qpos)
        _failed = failed(self._resting_block_height, self._goal_block_height(qpos))
        return self._compute_reward(goal_pos, should_grasp, gripper_pos,
                                    block_lifted, _failed)

    def step(self, action):
        action = np.clip(action * self._action_multiplier, -1, 1)

        mirrored = [
            'hand_l_proximal_motor',
            # 'hand_l_distal_motor'
        ]
        mirroring = [
            'hand_r_proximal_motor',
            # 'hand_r_distal_motor'
        ]

        def get_indexes(names):
            return [self.sim.name2id(ObjType.ACTUATOR, name) for name in names]

        # insert mirrored values at the appropriate indexes
        mirrored_indexes, mirroring_indexes = map(get_indexes, [mirrored, mirroring])
        # necessary because np.insert can't append multiple values to end:
        mirroring_indexes = np.minimum(mirroring_indexes, self.action_space.shape)
        action = np.insert(action, mirroring_indexes, action[mirrored_indexes])
        return super().step(action)
