import gym
import numpy as np
from gym import utils, spaces
from toy_environment.rectangle_object import RectangleObstacle
import pygame
import cv2
from toy_environment import room_obstacle_list, four_rooms_obstacle_list

class ContinuousGridworld2(gym.Env, utils.EzPickle):
    def __init__(self, obstacle_list_generator, visualize=False, image_size=64):
        utils.EzPickle.__init__(self, 'ContinuousGridworld2', 'image')
        self.observation_space = spaces.Box(-1, 1, shape=[4])
        self.action_space = spaces.Box(-1, 1, shape=[2])
        self.image_size = image_size
        self.obstacles = obstacle_list_generator(image_size)

        self.agent_position = self.get_non_intersecting_position()
        self.goal = self.get_non_intersecting_position()


        self.visualize = visualize
        if visualize:
            self.screen = pygame.display.set_mode((image_size, image_size),)
        else:
            self.screen = pygame.Surface((image_size, image_size))

        self.max_action_step = 0.2
        self.dist_cutoff = 0.1
        self.max_time_steps = 1000
        self.time_step = 0


    def _step(self, action):
        action = self.preprocess_action(action)

        num_subchecks = 4
        subaction = 0.0
        for i in range(num_subchecks):
            intersects = self.check_intersects(self.agent_position, action, mult=i / float(num_subchecks))
            if intersects:
                break
            else:
                subaction = action * (i / float(num_subchecks))


        self.agent_position = np.clip(self.agent_position + subaction, -1, 1)
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
        self.agent_position = self.get_non_intersecting_position()
        self.goal = self.get_non_intersecting_position()
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


    #### Hindsight Stuff

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


    ### Rendering

    def render_agent(self):
        x = (self.agent_position[0] + 1) / 2.
        y = (self.agent_position[1] + 1) / 2.
        self.screen.fill((255,255,255))
        x_int = int(x*self.image_size)
        y_int = int(y*self.image_size)
        pygame.draw.circle(self.screen, (0, 0, 0), (x_int, y_int), 3)
        for obs in self.obstacles:
            obs.draw(self.screen, (0,0,0))
        if self.visualize:
            pygame.display.update()
            pygame.event.get()
        imgdata = pygame.surfarray.array3d(self.screen)
        imgdata.swapaxes(0,1)
        return imgdata

    ### Collision Handling

    def get_non_intersecting_position(self):
        intersects = True
        while intersects:
            intersects = False
            position = np.random.uniform(-1, 1, size=2)
            tl = self.image_size * (position + 1) / 2. - 0.5 * (5 / np.sqrt(2))
            agent_rect = pygame.Rect(tl[0], tl[1], 5 / np.sqrt(2), 5 / np.sqrt(2))
            for obstacle in self.obstacles:
                collision = obstacle.collides(agent_rect)
                intersects |= collision
        return position

    def check_intersects(self, agent_position, scaled_action, mult=1.0):
        position = np.clip(agent_position + mult*scaled_action, -1, 1)
        intersects = False

        tl = (self.image_size*(position + 1)) / 2. - 0.5 * 0.01
        agent_rect = pygame.Rect(tl[0], tl[1], 0.01, 0.01)
        print(tl[0], tl[1], 0.01, 0.01)
        for obstacle in self.obstacles:
            collision = obstacle.collides(agent_rect)
            #print(obstacle.rect.topleft, obstacle.rect.width, obstacle.rect.height)
            intersects |= collision
            if collision:
                obstacle.color = (255, 0, 0)
            else:
                obstacle.color = (0, 0, 0)
            if intersects:
                break
        print(intersects)
        return intersects


if __name__ == '__main__':
    env = ContinuousGridworld2(four_rooms_obstacle_list.obstacle_list)
    obs = env.reset()
    print('pos:', obs[:2], 'goal:', obs[2:])
    while True:
        action = {'w': [1.0, 0],
                  's': [-1.0, 0],
                  'a': [0, 1.0],
                  'd': [0, -1.0]}.get(input('action:'), [0.0, 0.0])
        #action = np.random.uniform(-1, 1, size=2)
        obs, reward, terminal, info = env.step(action)
        image = env.render_agent()
        cv2.imshow('game', image)
        cv2.waitKey(1)
        print('pos:', obs[:2], 'goal:', obs[2:], 'reward:', reward)
        if terminal:
            env.reset()

