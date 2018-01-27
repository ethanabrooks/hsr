from os.path import join, dirname

import numpy as np
from gym import spaces
from mujoco import ObjType
import mujoco

from environment.base import BaseEnv, at_goal, print1


def failed(resting_block_height, goal_block_height):
    return False
    # return resting_block_height - goal_block_height > .02  #.029


class PickAndPlaceMocapEnv(BaseEnv):
    def __init__(self, max_steps, use_camera, geofence=.05, tb_dir=None, neg_reward=True, history_len=4,
                 image_dimensions=(64, 64), action_multiplier=1, gripper_mocap_dof=3):

        self._dimensions = 64, 64
        if use_camera:
            observation_space = spaces.Box(
                0, 1, shape=(list(self._dimensions) + [3 * history_len]))
        else:
            observation_space = spaces.Box(-1, 1, shape=(4 * history_len + 2))

        self._action_multiplier = action_multiplier
        self._history_len = history_len
        self._goal_block_name = 'block1'
        self.min_lift_height = 0.02
        self._resting_block_height = .49  # empirically determined
        left_finger_name = 'hand_l_distal_link'
        self._finger_names = [left_finger_name, left_finger_name.replace('_l_', '_r_')]

        super().__init__(
            geofence=geofence,
            max_steps=max_steps,
            observation_space=observation_space,
            xml_filepath=join('models', 'pick-and-place', 'world_mocap.xml'),
            history_len=history_len,
            tb_dir=tb_dir,
            use_camera=use_camera,
            neg_reward=neg_reward,
            body_name="hand_palm_link",
            steps_per_action=10,
            image_dimensions=image_dimensions)

        self.action_space = spaces.Box(-1, 1, shape=self.sim.nu + gripper_mocap_dof)
        self._table_height = self.sim.get_body_xpos('pan')[2]
        self._prev_quat = self.sim.mocap_quat

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
        return self._goal_block_height() - self._resting_block_height > self.min_lift_height

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
        goal_pos, should_grasp = self._destructure_goal(goal)
        qpos, block_lifted = self._destructure_obs(mlp_input)
        gripper_pos = self._gripper_pos(qpos)
        _failed = failed(self._resting_block_height, self._goal_block_height(qpos))
        return self._compute_reward(goal_pos, should_grasp, gripper_pos,
                                    block_lifted, _failed)

    def euler2quat(self, angle):
        """ Computes the quaternion when the z-rotation is given in degrees"""
        roll = pitch = 0.001
        yaw = angle

        cy = np.cos(yaw * 0.5);
        sy = np.sin(yaw * 0.5);
        cr = np.cos(roll * 0.5);
        sr = np.sin(roll * 0.5);
        cp = np.cos(pitch * 0.5);
        sp = np.sin(pitch * 0.5);

        qw = cy * cr * cp + sy * sr * sp;
        qx = cy * sr * cp - sy * cr * sp;
        qy = cy * cr * sp + sy * sr * cp;
        qz = sy * cr * cp - cy * sr * sp;
       
        return np.array([qw, qx, qy, qz])

    def set_current_quat(self):
        return self._prev_quat

    def step(self, action):
        # Last three items are desired gripper pos
        angle = action[1]
        mocap_pos = action[2:]
       
        # first two inputs are control:
        # action[0] = desired distance betwen grippers
        # mirroring l / r gripper
        action = [action[0], action[0]]

        # action[1] = desired rotation
        quat = self.euler2quat(angle)
        # print(action[1], quat)

        # print(action, mocap_pos)
        # Split ctrl and mocap
        if not np.all(mocap_pos == 0.0):
            self.sim.mocap_pos[0:3] = mocap_pos
            print('updating', debug)

        if float(angle) != 0.0 and (not np.all(quat == self._prev_quat)):
            self.sim.mocap_quat[0:4] = quat
            self._prev_quat = quat
            print('Mocap', self.sim.mocap_pos)
            print('updating quat', float(angle) != 0.0, (not np.all(quat == self._prev_quat)))

        return super().step(action)
