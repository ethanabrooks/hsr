import gym
import numpy as np
from gym import utils, spaces


class ContinuousGridworld2(gym.Env, utils.EzPickle):

    def __init__(self):
        utils.EzPickle.__init__(self, 'ContinuousGridworld2', 'image')
        self.observation_space = spaces.Box(-1, 1, shape=[4])
        self.action_space = spaces.Box(-1, 1, shape=[2])
        self.agent_position = self.sample_position()
        self.goal = self.sample_position()

        self.max_action_step = 0.2
        self.dist_cutoff = 0.1
        self.max_time_steps = 1000
        self.time_step = 0


    def _step(self, action):
        action = self.preprocess_action(action)
        self.agent_position = np.clip(self.agent_position + action, -1, 1)
        self.time_step += 1
        obs = self.obs()
        terminal = self.at_goal(self.goal, obs)
        if terminal:
            print('AT GOAL')
        reward = self.compute_reward(self.goal, obs)
        if self.time_step >= self.max_time_steps:
            terminal = True
        return obs, reward, terminal, {}


    def _reset(self):
        self.agent_position = self.sample_position()
        self.goal = self.sample_position()
        self.time_step = 0
        return self.obs()


    def obs(self):
        return np.concatenate([self.agent_position, self.goal], axis=0)


    def sample_position(self):
        return np.random.uniform(-1, 1, size=[2])


    def preprocess_action(self, action):
        action = np.array(action)
        # rescale the action to be between 0 and 1.
        radius = np.sqrt(action[0] ** 2 + action[1] ** 2)
        if radius > 1:
            action = action / radius
        action = action * self.max_action_step
        return action


    def at_goal(self, goal, obs):
        without_goal = obs[:-2]
        dist = np.sqrt(np.sum(np.square(without_goal - goal)))
        return dist <= self.dist_cutoff


    def compute_new_obs(self, goal, obs):
        without_goal = obs[:-2]
        return np.concatenate([without_goal, goal], axis=0)


    def compute_reward(self, goal, obs):
        return 1.0 if self.at_goal(goal, obs) else -0.01


    def obs_to_goal(self, obs):
        return obs[:2]

if __name__ == '__main__':
    env = ContinuousGridworld2()
    obs = env.reset()
    print('pos:', obs[:2], 'goal:', obs[2:])
    while True:
        #action = {'w': [1.0, 0],
        #          's': [-1.0, 0],
        #          'a': [0, 1.0],
        #          'd': [0, -1.0]}.get(input('action:'), [0.0, 0.0])
        action = np.random.uniform(-1, 1, size=2)
        obs, reward, terminal, info = env.step(action)
        print('pos:', obs[:2], 'goal:', obs[2:], 'reward:', reward)
        if terminal:
            env.reset()
