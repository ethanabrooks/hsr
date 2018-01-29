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

        self._goal_block_name = 'block1'
        self._resting_block_height = .428  # empirically determined
        self._min_lift_height = 0.02

        self._dimensions = 64, 64
        if use_camera:
            observation_space = spaces.Box(
                0, 1, shape=(list(self._dimensions) + [3 * history_len]))
        else:
            observation_space = spaces.Box(-1, 1, shape=(4 * history_len + 2))

        super().__init__(
            geofence=geofence,
            max_steps=max_steps,
            xml_filepath=join('models', 'pick-and-place', 'world_mocap.xml'),
            history_len=history_len,
            use_camera=False,
            neg_reward=neg_reward,
            body_name="hand_palm_link",
            steps_per_action=10,
            image_dimensions=None)

        self._action_multiplier = action_multiplier
        left_finger_name = 'hand_l_distal_link'
        self._finger_names = [left_finger_name, left_finger_name.replace('_l_', '_r_')]
        obs_size = history_len * sum(map(np.size, self._obs())) + sum(map(np.size, self._goal()))
        assert obs_size != 0
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs_size)
        self.action_space = spaces.Box(-1, 1, shape=self.sim.nu - 1)
        self._table_height = self.sim.get_body_xpos('pan')[2]

    # goal stuff

    def reset_qpos(self):
        self.init_qpos[3:7] = np.random.random(4)
        return self.init_qpos

    def _set_new_goal(self):
        pass

    def _obs(self):
        return self.sim.qpos, [self._block_lifted()]

    def _block_lifted(self):
        x, y, z = self.sim.get_body_xpos(self._goal_block_name)
        block_lifted = z - self._resting_block_height > self._min_lift_height
        return block_lifted

    def _goal(self):
        return self.sim.get_body_xpos(self._goal_block_name), [True]

    def goal_3d(self):
        return self._goal()[0]

    def _currently_failed(self):
        return False

    def _compute_terminal(self, goal, obs):
        goal, should_lift = goal
        qpos, block_lifted = obs
        return at_goal(self._gripper_pos(qpos), goal, self._geofence) and should_lift == block_lifted

    def _compute_reward(self, goal, obs):
        goal_pos, should_lift = goal
        qpos, block_lifted = obs
        if at_goal(self._gripper_pos(qpos), goal_pos, self._geofence) and block_lifted == should_lift:
            return 1
        elif self._neg_reward:
            return -.0001
        else:
            return 0

    def _obs_to_goal(self, obs):
        qpos, block_lifted = obs
        should_lift = block_lifted
        return self._gripper_pos(qpos), should_lift

    def _gripper_pos(self, qpos=None):
        finger1, finger2 = [self.sim.get_body_xpos(name, qpos)
                            for name in self._finger_names]
        return (finger1 + finger2) / 2.

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


    def step(self, action):
        # Last three items are desired gripper pos
        angle = action[1]
        mocap_pos = action[2:]
       
        # first two inputs are control:
        # action[0] = desired distance betwen grippers
        # mirroring l / r gripper

        # action = [wrist_roll, l_finger, r_finger]
        action = [0, action[0], action[0]]

        # Split ctrl and mocap
        if not np.all(mocap_pos == 0.0):
            self.sim.mocap_pos[0:3] = mocap_pos

        return super().step(action)
