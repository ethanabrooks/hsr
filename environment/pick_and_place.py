from os.path import join

import numpy as np
from gym import spaces
from mujoco import ObjType

from environment.base import BaseEnv, at_goal, print1


def failed(resting_block_height, goal_block_height):
    return False
    # return resting_block_height - goal_block_height > .02  #.029


class PickAndPlaceEnv(BaseEnv):
    def __init__(self, max_steps, geofence=.06, neg_reward=True, history_len=1):
        self._goal_block_name = 'block1'
        self._resting_block_height = .428  # empirically determined
        self._min_lift_height = 0.02

        super().__init__(
            geofence=geofence,
            max_steps=max_steps,
            xml_filepath=join('models', 'pick-and-place', 'world.xml'),
            history_len=history_len,
            use_camera=False,
            neg_reward=neg_reward,
            body_name="hand_palm_link",
            steps_per_action=10,
            image_dimensions=None)

        left_finger_name = 'hand_l_distal_link'
        self._finger_names = [left_finger_name,
                              left_finger_name.replace('_l_', '_r_')]
        obs_size = history_len * sum(map(np.size, self._obs())) + sum(
            map(np.size, self._goal()))
        assert obs_size != 0
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs_size)
        self.action_space = spaces.Box(-1, 1, shape=self.sim.nu - 1)
        self._table_height = self.sim.get_body_xpos('pan')[2]
        self._rotation_actuators = ["arm_flex_motor", "wrist_roll_motor"]

        # self._n_block_orientations = n_orientations = 8
        # self._block_orientations = np.random.uniform(0, 2 * np.pi,
                                                     # size=(n_orientations, 4))
        # self._rewards = np.ones(n_orientations) * -np.inf
        # self._usage = np.zeros(n_orientations)
        # self._current_orienation = None

    def reset_qpos(self):
        block_joint = self.sim.jnt_qposadr('block1joint')
        # self.init_qpos[block_joint + 3:block_joint + 7] = np.random.random(
        #     4) * 2 * np.pi
        self.init_qpos[block_joint + 3] = np.random.uniform(0, 1)
        self.init_qpos[block_joint + 6] = np.random.uniform(-1, 1)
        # mean_rewards = self._rewards / np.maximum(self._usage, 1)
        # self._current_orienation = i = np.argmin(mean_rewards)
        # print('rewards:', mean_rewards, 'argmin:', i)
        # self._usage[i] += 1
        # self.init_qpos[block_joint + 3:block_joint + 7] = self._block_orientations[i]
        self.init_qpos[self.sim.jnt_qposadr(
            'wrist_roll_joint')] = np.random.random() * 2 * np.pi
        return self.init_qpos

    def _set_new_goal(self):
        pass

    def _obs(self):
        return self.sim.qpos, [self._fingers_touching(), self._block_lifted()]

    def _fingers_touching(self):
        return not np.allclose(self.sim.sensordata[1:], [0, 0], atol=1e-2)

    def _block_lifted(self):
        return np.allclose(self.sim.sensordata[:1], [0], atol=1e-2)

    def _block_lifted2(self):
        return self._fingers_touching() and self._block_lifted()

    def _goal(self):
        return self.sim.get_body_xpos(self._goal_block_name), [True]

    def goal_3d(self):
        return self._goal()[0]

    def _currently_failed(self):
        return False

    def _achieved_goal(self, goal, obs):
        goal_pos, (should_lift,) = goal
        qpos, (fingers_touching, block_lifted) = obs
        _at_goal = at_goal(self._gripper_pos(qpos), goal_pos, self._geofence)
        result = _at_goal and should_lift == (block_lifted and fingers_touching)
        assert result == self._achieved_goal2(goal, (qpos, [self._block_lifted2()]))
        return result

    def _achieved_goal2(self, goal, obs):
        goal, (should_lift,) = goal
        qpos, (block_lifted,) = obs
        _at_goal = at_goal(self._gripper_pos(qpos), goal, self._geofence)
        result = _at_goal and should_lift == block_lifted
        return result

    def _compute_terminal(self, goal, obs):
        return self._achieved_goal(goal, obs)

    def _compute_reward(self, goal, obs):
        if self._achieved_goal(goal, obs):
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

    def step(self, action):
        action = np.clip(action, -1, 1)
        for name in self._rotation_actuators:
            i = self.sim.name2id(ObjType.ACTUATOR, name)
            action[i] *= np.pi / 2

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
        mirrored_indexes, mirroring_indexes = map(get_indexes,
                                                  [mirrored, mirroring])
        # necessary because np.insert can't append multiple values to end:
        mirroring_indexes = np.minimum(mirroring_indexes,
                                       self.action_space.shape)
        action = np.insert(action, mirroring_indexes, action[mirrored_indexes])
        return super().step(action)

