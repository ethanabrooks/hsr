import pygame
import gym
from gym import utils, spaces
import numpy as np
import sys
import cv2

class RectangleObstacle(object):

    def __init__(self, image_size, color, top_left, bottom_right):
        self.color = color
        self.tl = top_left
        self.br = bottom_right
        self.rect = pygame.Rect(image_size*self.tl[0], image_size*self.tl[1], image_size*(self.br[0] - self.tl[0]), image_size*(self.br[1] - self.tl[1]))
        self.ims = image_size

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (self.ims*self.tl[0], self.ims*self.tl[1], self.ims*(self.br[0] - self.tl[0]), self.ims*(self.br[1] - self.tl[1])))


    # vector from the agent in the direction that it wants to go
    # normalized by length 1
    def get_lines(self, image_size):
        ims = image_size
        lines = []
        # left vertical
        y1, y2 = self.tl[1], self.br[1]
        x1, x2 = self.tl[0], self.tl[0]
        lines.append((ims*y1, ims*y2, ims*x1, ims*x2))

        # right vertical
        y1, y2 = self.tl[1], self.br[1]
        x1, x2 = self.br[0], self.br[0]
        lines.append((ims*y1, ims*y2, ims*x1, ims*x2))

        # top horizontal
        y1, y2 = self.tl[1], self.tl[1]
        x1, x2 = self.tl[0], self.br[0]
        lines.append((ims*y1, ims*y2, ims*x1, ims*x2))

        # bottom horizontal
        y1, y2 = self.br[1], self.br[1]
        x1, x2 = self.tl[0], self.br[0]
        lines.append((ims*y1, ims*y2, ims*x1, ims*x2))

        return lines

    def compute_intersection(self, image_size, xs, ys, xa, ya):
        lines = self.get_lines(image_size)
        valid_ts = [1.0]
        for (y1, y2, x1, x2) in lines:
            if x1 == x2:
                t = (x1 - xs)/xa
            elif y1 == y2:
                t = (y1 - xs)/ya
            else:
                LHS = (xs - x1)/float(x2 - x1) - (ys - y1)/float(y2 - y1)
                RHS_coeff = ya/float(y2 - y1) - xa/float(x2 - x1)
                t = LHS / RHS_coeff
            if 1 > t >= 0:
                valid_ts.append(t)

        return valid_ts

    def collides(self, agent_rect):
        return self.rect.colliderect(agent_rect)

raw_obs_list_small_house = [
    (np.array([-0.5,  1.6]), np.array([ 0.05,  0.4 ])),
    (np.array([-0.5, -0.1]), np.array([ 0.05,  0.4 ])),
    #(np.array([-0.5 ,  0.75]), np.array([ 0.05,  0.45])),
    (np.array([-2.,  0.]), np.array([ 0.05,  2.1 ])),
    (np.array([ 2.,  0.]), np.array([ 0.05,  2.1 ])),
    (np.array([ 0.,  2.]), np.array([ 2.1 ,  0.05])),
    (np.array([ 0., -2.]), np.array([ 2.1 ,  0.05])),
    (np.array([ 2.,  2.]), np.array([ 0.1,  0.2])),
    (np.array([-2.,  2.]), np.array([ 0.2,  0.1])),
    (np.array([ 2., -2.]), np.array([ 0.2,  0.1])),
    (np.array([-2., -2.]), np.array([ 0.1,  0.2])),
    (np.array([ 0.75,  1.6 ]), np.array([ 1.25,  0.4 ])),
    (np.array([ 0.4,  0.2]), np.array([ 0.03,  0.03])),
    (np.array([-0.4,  0.2]), np.array([ 0.03,  0.03])),
    (np.array([ 0.4, -0.2]), np.array([ 0.03,  0.03])),
    (np.array([-0.4, -0.2]), np.array([ 0.03,  0.03])),
    (np.array([ 0.,  0.]), np.array([ 0.5,  0.3])),
    (np.array([ 1.4,  1.5]), np.array([ 0.05,  0.05])),
    (np.array([ 0.00724405,  0.00047065]),
     np.array([ 0.12458548,  0.22737432])),
    (np.array([-0.07262482,  0.00042312]),
     np.array([ 0.14067157,  0.18287495])),
    (np.array([ 0.01973567, -0.01924405]),
     np.array([ 0.03065044,  0.06756446])),
    (np.array([-0.03566579,  0.00042771]),
     np.array([ 0.23748511,  0.22563542])),
    (np.array([ 0.,  0.]),
     np.array([ 0.04 ,  0.015])),
    (np.array([ 0.,  0.]),
     np.array([ 0.04 ,  0.015])),
    (np.array([ 0.,  0.]),
     np.array([ 0.04 ,  0.015])),
    (np.array([ 0.,  0.]),
     np.array([ 0.04 ,  0.015])),
    (np.array([ 0.,  0.]),
     np.array([ 0.03,  0.  ])),
    (np.array([ 0.,  0.]), np.array([ 0.03,  0.  ])),
    (np.array([ 0.,  0.]), np.array([ 0.03,  0.  ])),
    (np.array([ 0.,  0.]), np.array([ 0.03,  0.  ])),
    (np.array([  2.99617112e-10,  -1.40730885e-10]),
     np.array([ 0.0202575,  0.0202575])),
    (np.array([  2.99617112e-10,  -1.40730885e-10]),
     np.array([ 0.0202575,  0.0202575])),
    (np.array([-0.039093  , -0.00035366]),
     np.array([ 0.07135277,  0.08104951])),
    (np.array([-0.03874661, -0.00035598]),
     np.array([ 0.06920656,  0.07723567])),
    (np.array([  2.21389409e-03,  -3.13193576e-05]),
     np.array([ 0.04224088,  0.06793738])),
    (np.array([  2.21389409e-03,  -3.13193576e-05]),
     np.array([ 0.04224088,  0.06793738])),
    (np.array([  5.49594413e-03,   1.64354635e-05]),
     np.array([ 0.02540056,  0.07078958])),
    (np.array([ -3.28302671e-02,   1.83681429e-05]),
     np.array([ 0.06936514,  0.08405513])),
    (np.array([-0.09445665, -0.00024871]),
     np.array([ 0.02890534,  0.09901419])),
    (np.array([ -5.54383682e-02,   3.47860699e-05]),
     np.array([ 0.07262519,  0.10925511])),
    (np.array([-0.07627559, -0.00039322]),
     np.array([ 0.0043356 ,  0.05954185])),
    (np.array([  1.18795196e-04,  -2.67583751e-05]),
     np.array([ 0.02127834,  0.02137586])),
    (np.array([  1.18795196e-04,  -2.67583751e-05]),
     np.array([ 0.02127834,  0.02137586])),
    (np.array([  2.20022990e-02,  -1.07730592e-08]),
     np.array([ 0.0124997 ,  0.01824984]))]

raw_obs_list_big_house = [
    (np.array([-0.5, -0.5]), np.array([ 0.1,  0.5])),
    (np.array([-0.5, -5. ]), np.array([ 0.1,  1. ])),
    (np.array([-1.5, -0.5]), np.array([ 1. ,  0.1])),
    (np.array([-5. , -0.5]), np.array([ 1. ,  0.1])),
    (np.array([-6.,  0.]), np.array([ 0.01,  6.  ])),
    (np.array([ 6.,  0.]), np.array([ 0.01,  6.  ])),
    (np.array([ 0.,  6.]), np.array([ 6.  ,  0.01])),
    (np.array([ 0., -6.]), np.array([ 6.  ,  0.01])),
    (np.array([-5.03557879, -3.00072784]), np.array([ 0.55392665,  0.8984974 ])),
    (np.array([-5.25, -5.25]), np.array([ 0.5,  0.5])),
    (np.array([ 2.75,  5.55]), np.array([ 1.25,  0.4 ])),
    (np.array([ 4.75, -5.55]), np.array([ 1.25,  0.4 ])),
    (np.array([ 3.1,  3.3]), np.array([ 0.03,  0.03])),
    (np.array([ 4.1,  3.3]), np.array([ 0.03,  0.03])),
    (np.array([ 4.1,  1.7]), np.array([ 0.03,  0.03])),
    (np.array([ 3.1,  1.7]), np.array([ 0.03,  0.03])),
    (np.array([ 3.5,  2.5]), np.array([ 0.75,  1.  ])),
    (np.array([-5.5,  5.5]), np.array([ 0.45,  0.25])),
    (np.array([-3.,  3.]), np.array([ 1.  ,  0.03])),
    (np.array([-3.5,  3.5]), np.array([ 0.03,  0.03])),
    (np.array([-3.5,  2.5]), np.array([ 0.03,  0.03])),
    (np.array([-2.5,  3.5]), np.array([ 0.03,  0.03])),
    (np.array([-2.5,  2.5]), np.array([ 0.03,  0.03])),
    (np.array([ 2.9, -2.7]), np.array([ 0.03,  0.03])),
    (np.array([ 2.9, -2.3]), np.array([ 0.03,  0.03])),
    (np.array([ 2.1, -2.7]), np.array([ 0.03,  0.03])),
    (np.array([ 2.1, -2.3]), np.array([ 0.03,  0.03])),
    (np.array([ 2.5, -2.5]), np.array([ 0.75 ,  0.375])),
    (np.array([ 4.36981214,  2.50113901]), np.array([ 0.23128587,  0.329117  ])),
    (np.array([ 4.4 ,  2.75]), np.array([ 0.025,  0.2  ])),
    (np.array([ 4.1 ,  2.75]), np.array([ 0.025,  0.2  ])),
    (np.array([ 4.4 ,  2.25]), np.array([ 0.025,  0.2  ])),
    (np.array([ 4.1 ,  2.25]), np.array([ 0.025,  0.2  ])),
    (np.array([ 2.38018786,  2.49886099]), np.array([ 0.23128587,  0.329117  ])),
    (np.array([ 2.35,  2.75]), np.array([ 0.025,  0.2  ])),
    (np.array([ 2.35,  2.25]), np.array([ 0.025,  0.2  ])),
    (np.array([ 2.65,  2.75]), np.array([ 0.025,  0.2  ])),
    (np.array([ 2.65,  2.25]), np.array([ 0.025,  0.2  ])),
    (np.array([ 0.00724405,  0.00047065]), np.array([ 0.12458548,  0.22737432])),
    (np.array([-0.07262482,  0.00042312]), np.array([ 0.14067157,  0.18287495])),
    (np.array([ 0.01973567, -0.01924405]), np.array([ 0.03065044,  0.06756446])),
    (np.array([-0.03566579,  0.00042771]), np.array([ 0.23748511,  0.22563542])),
    (np.array([ 0.,  0.]), np.array([ 0.04 ,  0.015])),
    (np.array([ 0.,  0.]), np.array([ 0.04 ,  0.015]))
]

'''raw_obs_list_big_house = [
    (np.array([-0.5,  2. ]), np.array([ 0.1,  4. ])),
    (np.array([-0.5, -5. ]), np.array([ 0.1,  1. ])),
    #(np.array([-0.5, -2.5]), np.array([ 0.1,  1.5])),
    (np.array([-1.5, -0.5]), np.array([ 1. ,  0.1])),
    (np.array([-5. , -0.5]), np.array([ 1. ,  0.1])),
    #(np.array([-2.5, -0.5]), np.array([ 2. ,  0.1])),
    (np.array([-6.,  0.]), np.array([ 0.01,  6.  ])),
    (np.array([ 6.,  0.]), np.array([ 0.01,  6.  ])),
    (np.array([ 0.,  6.]), np.array([ 6.  ,  0.01])),
    (np.array([ 0., -6.]), np.array([ 6.  ,  0.01])),
    (np.array([-5.03557879, -3.00072784]), np.array([ 0.55392665,  0.8984974 ])),
    (np.array([-5.25, -5.25]), np.array([ 0.5,  0.5])),
    (np.array([ 2.75,  5.55]), np.array([ 1.25,  0.4 ])),
    (np.array([ 4.75, -5.55]), np.array([ 1.25,  0.4 ])),
    (np.array([ 3.1,  3.3]), np.array([ 0.03,  0.03])),
    (np.array([ 4.1,  3.3]), np.array([ 0.03,  0.03])),
    (np.array([ 4.1,  1.7]), np.array([ 0.03,  0.03])),
    (np.array([ 3.1,  1.7]), np.array([ 0.03,  0.03])),
    (np.array([ 3.5,  2.5]), np.array([ 0.75,  1.  ])),
    (np.array([-5.5,  5.5]), np.array([ 0.45,  0.25])),
    (np.array([-3.,  3.]), np.array([ 1.  ,  0.03])),
    (np.array([-3.5,  3.5]), np.array([ 0.03,  0.03])),
    (np.array([-3.5,  2.5]), np.array([ 0.03,  0.03])),
    (np.array([-2.5,  3.5]), np.array([ 0.03,  0.03])),
    (np.array([-2.5,  2.5]), np.array([ 0.03,  0.03])),
    (np.array([ 2.9, -2.7]), np.array([ 0.03,  0.03])),
    (np.array([ 2.9, -2.3]), np.array([ 0.03,  0.03])),
    (np.array([ 2.1, -2.7]), np.array([ 0.03,  0.03])),
    (np.array([ 2.1, -2.3]), np.array([ 0.03,  0.03])),
    (np.array([ 2.5, -2.5]), np.array([ 0.75 ,  0.375])),
    (np.array([ 4.36981214,  2.50113901]), np.array([ 0.23128587,  0.329117  ])),
    (np.array([ 4.4 ,  2.75]), np.array([ 0.025,  0.2  ])),
    (np.array([ 4.1 ,  2.75]), np.array([ 0.025,  0.2  ])),
    (np.array([ 4.4 ,  2.25]), np.array([ 0.025,  0.2  ])),
    (np.array([ 4.1 ,  2.25]), np.array([ 0.025,  0.2  ])),
    (np.array([ 2.38018786,  2.49886099]), np.array([ 0.23128587,  0.329117  ])),
    (np.array([ 2.35,  2.75]), np.array([ 0.025,  0.2  ])),
    (np.array([ 2.35,  2.25]), np.array([ 0.025,  0.2  ])),
    (np.array([ 2.65,  2.75]), np.array([ 0.025,  0.2  ])),
    (np.array([ 2.65,  2.25]), np.array([ 0.025,  0.2  ])),
    (np.array([ 0.00724405,  0.00047065]), np.array([ 0.12458548,  0.22737432])),
    (np.array([-0.07262482,  0.00042312]), np.array([ 0.14067157,  0.18287495])),
    (np.array([ 0.01973567, -0.01924405]), np.array([ 0.03065044,  0.06756446])),
    (np.array([-0.03566579,  0.00042771]), np.array([ 0.23748511,  0.22563542]))]'''

def process_obstacle_small_house(pos, size):
    pos = (pos + 2) / 4.
    size = size / 4.
    tl = [pos[0]-size[0], pos[1]-size[1]]
    br = [pos[0]+size[0], pos[1]+size[1]]
    return tl, br

def process_obstacle_big_house(pos, size):
    pos = (pos + 6) / 12.
    size = size / 12.
    tl = [pos[0]-size[0], pos[1]-size[1]]
    br = [pos[0]+size[0], pos[1]+size[1]]
    return tl, br

def build_obs_list_small_house(image_size, color):
    obs_list = []
    for pos, size in raw_obs_list_small_house:
        tl, br = process_obstacle_small_house(pos, size)
        obs_list.append(RectangleObstacle(image_size, color, tl, br))
    return obs_list

def build_obs_list_big_house(image_size, color):
    obs_list = []
    for pos, size in raw_obs_list_big_house:
        tl, br = process_obstacle_big_house(pos, size)
        obs_list.append(RectangleObstacle(image_size, color, tl, br))
    return obs_list

NO_OBJECTS, SMALL_HOUSE, BIG_HOUSE = range(3)

class ContinuousGridworld(gym.Env, utils.EzPickle):
    """ The environment """

    def __init__(self, game, use_cnn=False, image_size=64, history_length=1, visualize=False, init_goal_coord=(0.5,0), max_steps=6000, obstacle_mode=NO_OBJECTS):
        assert obstacle_mode in [NO_OBJECTS, SMALL_HOUSE, BIG_HOUSE]
        utils.EzPickle.__init__(self, game, 'image')
        self.image_size = image_size
        self.visualize = visualize
        self.use_cnn = use_cnn
        self.closeness_cutoff = 0.3
        if visualize:
            self.screen = pygame.display.set_mode((image_size, image_size),)
        else:
            self.screen = pygame.Surface((image_size, image_size))
        pygame.init()
        self._goal = np.array(list(init_goal_coord))
        self.agent_position = np.array([-0.5, -0.5])
        self.history_length = history_length
        if self.use_cnn:
            self.observation_space = spaces.Box(0, 1, shape=(image_size, image_size, 3*self.history_length))
        else:
            self.observation_space = spaces.Box(-1, 1, shape=(2*self.history_length + 2,))

        self.action_space = spaces.Box(-1, 1, shape=(2,))
        self.obstacles = []
        obstacle_color = (0, 0, 0)
        obstacle_builders = {
            NO_OBJECTS: lambda image_size, color: [],
            SMALL_HOUSE: build_obs_list_small_house,
            BIG_HOUSE: build_obs_list_big_house
        }

        self.obstacles = obstacle_builders[obstacle_mode](self.image_size, obstacle_color)

        #self.action_space = spaces.Discrete(4)
        self.render_agent()
        self.max_steps = max_steps
        self.obstacle_list = []
        self.step_counter = 0.
        self.buffer = ObsBuffer(self.history_length, 64)

    def normalize_coordinate(self, xy):
        return tuple(self.image_size*np.array(xy))


    def render_agent(self):
        x = (self.agent_position[0] + 1) / 2.
        y = (self.agent_position[1] + 1) / 2.
        self.screen.fill((255,255,255))
        x_int = int(x*self.image_size)
        y_int = int(y*self.image_size)
        pygame.draw.circle(self.screen, (0, 0, 0), (x_int, y_int), 3)
        for obs in self.obstacles:
            obs.draw(self.screen)
        if self.visualize:
            pygame.display.update()
            pygame.event.get()
        imgdata = pygame.surfarray.array3d(self.screen)
        imgdata.swapaxes(0,1)
        return imgdata

    def draw_target(self):
        x = (self._goal[0] + 1) / 2.
        y = (self._goal[1] + 1) / 2.
        x_int = int(x * self.image_size)
        y_int = int(y * self.image_size)
        pygame.draw.circle(self.screen, (255, 0, 255), (x_int, y_int), 5)
        if self.visualize:
            pygame.display.update()
            pygame.event.get()

    def obs(self):
        xy, img = self.buffer.get()
        if self.use_cnn:
            return xy, img
        else:
            return np.concatenate([self.agent_position, self.goal()], axis=0)


    @property
    def pos_buffer(self):
        xy, img = self.buffer.get()
        return xy

    @property
    def normalized_goal(self):
        return self._goal.copy()


    def check_intersects(self, agent_position, scaled_action, mult=1.0):
        position = np.clip(agent_position + mult*scaled_action, -1, 1)
        intersects = False

        tl = self.image_size * (position + 1) / 2. - 0.5 * (3 / np.sqrt(2))
        agent_rect = pygame.Rect(tl[0], tl[1], 0.01, 0.01)
        for obstacle in self.obstacles:
            collision = obstacle.collides(agent_rect)
            intersects |= collision
            if collision:
                obstacle.color = (255, 0, 0)
            else:
                obstacle.color = (0, 0, 0)
            if intersects:
                break
        print(intersects)
        return intersects

    def goal(self):
        return self._goal.copy()

    def _step(self, action):
        action = np.array(action)
        delta = 0.1 * 3 / 7. * 2

        # make sure action cannot have radius > 1.
        radius = np.sqrt(action[0]**2 + action[1]**2)
        if radius > 1:
            action = action / radius

        scaled_action = delta*action


        #forward, backward, turn_right, turn_left = range(4)

        self.step_counter += 1
        #if action == forward:
        #    scaled_action = np.array([delta*np.cos(self.agent_direction), delta*np.sin(self.agent_direction)])
        #elif action == backward:
        #    scaled_action = -np.array([delta*np.cos(self.agent_direction), delta*np.sin(self.agent_direction)])
        #elif action == turn_right:
        #    self.agent_direction = (self.agent_direction + delta) % (2*np.pi)
        #    scaled_action = np.array([0,0])
        #else:
        #    self.agent_direction = (self.agent_direction - delta) % (2*np.pi)
        #    scaled_action = np.array([0,0])


        num_subchecks = 4
        scaled_subaction = 0.0
        for i in range(num_subchecks):
            intersects = self.check_intersects(self.agent_position, scaled_action, mult=i/float(num_subchecks))
            if intersects:
                break
            else:
                scaled_subaction = scaled_action * (i/float(num_subchecks))


        if not intersects:
            self.agent_position = np.clip(self.agent_position + scaled_subaction, -1, 1)
        observation = self.render_agent()
        self.buffer.update(self.agent_position, observation.copy())
        self.draw_target()

        goal_dist = np.sqrt(np.sum(np.square(self.agent_position - self.goal())))
        #reward  = 1 if goal_dist < self.closeness_cutoff else -0.01
        reward = 1 if self.at_goal(self.goal(), self.obs()) else -0.01
        subtask_complete = False
        if reward == 1:
            print('AT GOAL')
            #self._goal = self.get_non_intersecting_position()
            subtask_complete = True
        if self.step_counter > self.max_steps:
            subtask_complete = True


        if self.use_cnn:
            raise Exception('You shouldnt be using this yet.')
            #return self.obs(), reward, subtask_complete, {}

        return self.obs(), reward, subtask_complete, {}


    def compute_reward(self, goal, xy):
        reward = 1 if self.at_goal(goal, xy) else -0.01
        return reward

    def compute_terminal(self, goal, xy):
        state_without_goal = xy[:-2]
        agent_position = state_without_goal[-2:]
        goal_dist = np.sqrt(np.sum(np.square(agent_position - goal)))
        return goal_dist < self.closeness_cutoff

    def compute_new_obs(self, goal, obs):
        state_without_goal = obs[:-2]
        return np.concatenate([state_without_goal, goal], axis=0)


    def obs_to_goal(self, obs):
        state_without_goal = obs[:-2]
        return state_without_goal[-2:]

    def at_goal(self, goal, obs):
        state_without_goal = obs[:-2]
        agent_position = state_without_goal[-2:]
        #print('agent_position', agent_position)
        goal_dist = np.sqrt(np.sum(np.square(agent_position - goal)))
        return goal_dist < self.closeness_cutoff

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

    def _reset(self):
        self.buffer.reset()
        self.agent_position = self.get_non_intersecting_position()
        self._goal = self.get_non_intersecting_position()
        self.buffer.update(self.agent_position, self.render_agent())
        self.step_counter = 0
        return self.obs()

    def _get_obs(self):
        return self.render_agent()




class ObsBuffer(object):
    def __init__(self, history_len, image_size):
        self.hl = history_len
        self.image_shape = [image_size, image_size, 3]
        self.reset()

    def update(self, xy, img):
        self.xy_buffer = self.xy_buffer[1:] + [xy]
        self.img_buffer = self.img_buffer[1:] + [img]

    def reset(self):
        self.xy_buffer = [np.zeros(2) for _ in range(self.hl)]
        self.img_buffer = [np.zeros(self.image_shape) for _ in range(self.hl)]

    def get(self):
        assert len(self.xy_buffer) == self.hl
        assert len(self.img_buffer) == self.hl
        xy = np.concatenate(self.xy_buffer, axis=0)
        img = np.concatenate([np.reshape(x, self.image_shape)
                              for x in self.img_buffer], axis=2)
        return xy, img


if __name__ == '__main__':
    move_map = {'w': 0,
                's': 1,
                'a': 2,
                'd': 3}
    grid = ContinuousGridworld('grid', visualize=True, obstacle_mode=NO_OBJECTS)
    running = True
    while running:

        #for event in pygame.event.get():
        #    if event.type == pygame.QUIT:
        #        running = False
        #        print
        #        "Exiting game."


        #pressed = pygame.key.get_pressed()
        #move = move_map.get(pressed, -1)
        #move = [move_map[key] for key in move_map if pressed[key]]
        key = input('Key:')
        move = move_map.get(key, None)
        if move is None:
            continue
        #move = np.random.randint(0, 4)
        continuous_map = {0: [1, 0], 1: [-1, 0], 2: [0, 1], 3: [0, -1]}
        obs, reward, done, info = grid.step(continuous_map[move])
        if done:
            grid.reset()

        print(obs, reward, done)



#print(grid.observation_space)
