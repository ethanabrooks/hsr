import gym
import numpy as np
from gym import utils, spaces
from toy_environment.rectangle_object import RectangleObstacle
import pygame
import cv2
from collections import deque
from toy_environment import room_obstacle_list, four_rooms_obstacle_list
import matplotlib.pyplot as plt

class ContinuousGridworld2(gym.Env, utils.EzPickle):

    def __init__(self, obstacle_list_generator, noise_type, use_cnn=False, visualize=False, image_size=64, max_action_step=0.2, max_time_steps=1000, eval_=False):   
        utils.EzPickle.__init__(self, 'ContinuousGridworld2', 'image')
        self.use_cnn = use_cnn
        self.eval = eval_

        if self.use_cnn:
            self.observation_space = spaces.Box(-1, 1, shape=[image_size, image_size, 4])
        else:
            self.observation_space = spaces.Box(-1, 1, shape=[4])

        self.action_space = spaces.Box(-1, 1, shape=[2])
        self.image_size = image_size
        self.obstacles = obstacle_list_generator(image_size, eval_=eval_)

        self.use_cnn = use_cnn
        self.agent_position = self.get_non_intersecting_position(self.agent_position_generator)
        self.goal = self.get_non_intersecting_position(self.goal_position_generator)
        self.visualize = visualize

        self.resolution = 0.025
        self.noise_type = noise_type
        height = 2.0 / self.resolution
        height = int(height)

        self.height = height
        self.achieved_goals = [np.zeros((height, height), dtype=int) for i in range(4)]
        self.missed_goals = [np.zeros((height, height), dtype=int) for i in range(4)]

        if visualize:
            self.screen = pygame.display.set_mode((image_size, image_size),)
        else:
            self.screen = pygame.Surface((image_size, image_size))

        self.max_action_step = max_action_step
        self.dist_cutoff = 0.2
        self.max_time_steps = max_time_steps
        self.time_step = 0

    def agent_position_generator(self):
        return np.random.uniform(-1, 1, size=2)

    def goal_position_generator(self):
        return np.random.uniform(-1, 1, size=2)

    def _step(self, action):
        action = self.preprocess_action(action)
        num_subchecks = 4
        for i in range(1, num_subchecks):
            intersects = self.check_intersects(self.agent_position, action, mult=i / float(num_subchecks))
            if intersects:
                action = action * (i-1) / float(num_subchecks)
                break
        self.agent_position = np.clip(self.agent_position + action, -1, 1)
        self.time_step += 1

        if self.time_step % 1000 == 0:
            self.generate_heatmap(filename='achieved_eval_{}'.format(self.eval))
            self.generate_heatmap(filename='missed_eval_{}'.format(self.eval))
        
        obs = self.obs()
        terminal = self.compute_terminal(self.goal, obs, xy=self.agent_position)
        reward = self.compute_reward(self.goal, obs, xy=self.agent_position)

        #cv2.imshow('game', self.render_agent())
        #cv2.waitKey(1)
        #if self.at_goal(self.goal, obs):
        #    self.goal = self.get_non_intersecting_position(self.goal_position_generator)
            #self.agent_position = self.get_non_intersecting_position(self.agent_position_generator)

        if reward == 1:
            x_goal = int((round(self.goal[0], 2) + 1) / self.resolution) - 1
            y_goal = int((round(self.goal[1], 2) + 1) / self.resolution) - 1
            x_goal = int(np.clip(x_goal, 0, self.height - 1))
            y_goal = int(np.clip(y_goal, 0, self.height - 1))
            self.achieved_goals[self.room][x_goal][y_goal] += 1
	
            print('AT GOAL')

        if self.time_step >= self.max_time_steps:
            if reward != 1:
                x_goal = int((round(self.goal[0], 2) + 1) / self.resolution) - 1
                y_goal = int((round(self.goal[1], 2) + 1) / self.resolution) - 1
                x_goal = int(np.clip(x_goal, 0, self.height - 1))
                y_goal = int(np.clip(y_goal, 0, self.height - 1))

                self.missed_goals[self.room][x_goal][y_goal] += 1
            
            terminal = True

        return obs, reward, terminal, {}


    def _reset(self):
        self.agent_position = self.get_non_intersecting_position(self.agent_position_generator)
        self.goal = self.get_non_intersecting_position(self.goal_position_generator)
        self.time_step = 0
        return self.obs()



    def obs(self):
        if self.use_cnn:
            screen = 2*(self.render_agent() / 255. - 0.5)
            goal = self.render_goal(self.goal.copy())
            return np.concatenate([screen, goal], axis=2)
        else:
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

    def at_goal(self, goal, obs, xy=None):
        if not self.use_cnn:
            without_goal = obs[:-2]
        else:
            if xy is None:
                without_goal = self.extract_position_from_image(obs)
            else:
                without_goal = xy
        dist = np.sqrt(np.sum(np.square(without_goal - goal)))
        return dist <= self.dist_cutoff


    def change_goal(self, goal, obs):
        if not self.use_cnn:
            without_goal = obs[:-2]
            return np.concatenate([without_goal, goal], axis=0)
        else:
            without_goal = obs[:, :, :3]
            goal = self.render_goal(goal)
            return np.concatenate([without_goal, goal], axis=2)


    def compute_reward(self, goal, obs, xy=None):
        return 1.0 if self.at_goal(goal, obs, xy=xy) else -0.01

    def compute_terminal(self, goal, obs, xy=None):
        return self.at_goal(goal, obs, xy=xy)


    def obs_to_goal(self, obs):
        if not self.use_cnn:
            return obs[:2]
        else:
            return self.extract_position_from_image(obs)



    ### Rendering

    def render_agent(self):
        x = (self.agent_position[0] + 1) / 2.
        y = (self.agent_position[1] + 1) / 2.
        self.screen.fill((255,255,255))
        x_int = int(x*self.image_size)
        y_int = int(y*self.image_size)
        x_goal, y_goal = (self.goal[0] + 1) / 2, (self.goal[1] + 1)/2
        x_goal_int = int(x_goal * self.image_size)
        y_goal_int = int(y_goal * self.image_size)
        pygame.draw.circle(self.screen, (255, 0, 0), (x_int, y_int), 3)
        #pygame.draw.circle(self.screen, (255, 0, 0), (x_goal_int, y_goal_int), 3)

        for obs in self.obstacles:
            obs.draw(self.screen, (0,255,0))
        if self.visualize:
            pygame.display.update()
            pygame.event.get()
        imgdata = pygame.surfarray.array3d(self.screen)
        #imgdata.swapaxes(0,1)
        return imgdata

    def render_goal(self, goal):
        x_goal, y_goal = (goal[0]) / 2, (goal[1] + 1) / 2
        x_goal_int = int(x_goal * self.image_size)
        y_goal_int = int(y_goal * self.image_size)
        pygame.draw.circle(self.screen, (0., 0., 0.), (x_goal_int, y_goal_int), 3)
        imgdata = pygame.surfarray.array3d(self.screen)
        return 2*(imgdata[:, :, [0]] / 255. - 0.5)
        #assert self.image_size % 2 == 0
        #return np.tile(np.reshape(goal, [2, 1, 1]), [self.image_size // 2, self.image_size, 1])

    def extract_position_from_image(self, image):
        raise Exception('Function has a bug. Dont use it.')
        space = np.linspace(-1, 1, num=self.image_size)
        X, Y = np.meshgrid(space, space)
        X = np.reshape(X, [self.image_size, self.image_size, 1])
        Y = np.reshape(Y, [self.image_size, self.image_size, 1])
        XY = np.concatenate([X, Y], axis=2)
        image = (image[:, :, [0]] + 1.0) / 2.
        prob = image / np.sum(image)
        mean_xy = np.mean((XY * prob), axis=(0,1))
        return mean_xy


    def generate_heatmap(self, filename):
        target = None
        for i in range(4):
            if filename is 'achieved':
                target = self.achieved_goals[i]
            else:
                target = self.missed_goals[i]

            plt.imsave('{}-noisy_pos-{}-noise-{}-room-{}.png'.format(filename, self.noisy_position, self.noise_type, i), target)

    ### Collision Handling

    def get_non_intersecting_position(self, generator):
        #if generator is None:
        #    generator = lambda: np.random.uniform(-1, 1, size=2)
        intersects = True
        while intersects:
            intersects = False
            position = generator()
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
        return intersects

class FourRoomExperiment(ContinuousGridworld2):

    def __init__(self, noise_type=None, visualize=False, noisy_position=False, image_size=64, use_cnn=False, eval_=False):
        from toy_environment import four_rooms_obstacle_list
        self.position_mapping = {0: [-0.5, -0.5], 1: [-0.5, 0.5], 2: [0.5, 0.5], 3: [0.5, -0.5]}
        self.noisy_position = noisy_position
        super().__init__(four_rooms_obstacle_list.obstacle_list, noise_type, visualize=visualize, image_size=image_size, use_cnn=use_cnn, eval_=eval_)
	
    def agent_position_generator(self):
        self.room = np.random.randint(0,4)
        pos = np.array(self.position_mapping[self.room])
        if self.noisy_position:
            pos = np.add(pos, np.random.uniform(low=-0.05, high=0.05, size=2))
        return pos 
 
    def goal_position_generator(self):
        goal = None
        if self.eval:
            goal = np.random.uniform(low=-1., high=1., size=2)
            return goal

        while True:
            goal = np.random.uniform(self.agent_position - 0.5, self.agent_position + 0.5)
            if goal[0] > -1 and goal[0] < 1 and goal[1] > -1 and goal[1] < 1:
                break

        return goal


class FourRoomDiscrete(FourRoomExperiment):
	def __init__(self, noise_type=None, visualize=False, noisy_position=False, image_size=84, use_cnn=True, eval_=False):
		super().__init__(noise_type, visualize, noisy_position, image_size, use_cnn=use_cnn, eval_=eval_)
		self.action_space = spaces.Discrete(4)
		self.action_mapping = [[1.0, 0], [-1.0, 0], [0, 1.0], [0, -1.0]]

	def _step(self, action):
		action = self.action_mapping[action]
		return super()._step(action)


if __name__ == '__main__':
    env = FourRoomDiscrete(visualize=False)
    # env = ContinuousGridworld2(room_obstacle_list.obstacle_list)
    obs = env.reset()
    print('pos:', obs[:2], 'goal:', obs[2:])

    while True:
        action = {'s': 0,
                  'w': 1,
                  'd': 2,
                  'a': 3}.get(input('action:'), None)
        #action = np.random.uniform(-1, 1, size=2)
        
        obs, reward, terminal, info = env.step(action)
        image = env.render_agent()
        cv2.imshow('game', image)
        cv2.waitKey(1)
        print('pos:', obs[:2], 'goal:', obs[2:], 'reward:', reward)
        if terminal:
            env.reset()

