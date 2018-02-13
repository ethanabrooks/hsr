from pygame import rect

import numpy as np
import pygame
from gym import spaces

from toy_environment import room_obstacle_list


def sample_position():
    return np.random.uniform(-1, 1, size=[2])


class ContinuousGridworld:
    def __init__(self, obstacle_list_generator, visualize=False, image_size=64, step_size=0.2,
                 max_time_steps=1000):
        self.observation_space = spaces.Box(-1, 1, shape=[4])
        self.action_space = spaces.Box(-1, 1, shape=[2])
        self.image_size = image_size
        self.obstacles = obstacle_list_generator(image_size)

        self.agent_position = self.get_non_intersecting_position(self.agent_position_generator)
        self.goal = self.get_non_intersecting_position(self.goal_position_generator)
        self.visualize = visualize
        if visualize:
            self.screen = pygame.display.set_mode((image_size, image_size), )
        else:
            self.screen = pygame.Surface((image_size, image_size))

        self.step_size = step_size
        self.dist_cutoff = 0.2
        self.max_steps = max_time_steps
        self.time_step = 0

        # required for OpenAI code
        self.metadata = {'render.modes': 'rgb_array'}
        self.reward_range = -np.inf, np.inf
        self.spec = None

    @staticmethod
    def seed(seed):
        np.random.seed(seed)

    def render(self, mode='human', close=False):
        if mode == 'rgb_array':
            return self.render_agent()
        elif mode == 'human':
            raise NotImplementedError
        else:
            raise RuntimeError('mode must be human|rgb_array')

    def agent_position_generator(self):
        return np.random.uniform(-1, 1, size=2)

    def goal_position_generator(self):
        return np.random.uniform(-1, 1, size=2)

    def step(self, action):
        assert isinstance(action, np.ndarray)

        # rescale the action
        radius = np.linalg.norm(action, ord=2)
        if radius > 1:
            action /= radius
        action *= self.step_size

        # TODO: I don't understand this
        num_subchecks = 4
        for i in range(1, num_subchecks):
            if self.check_intersects(self.agent_position,  # TODO
                                     action,
                                     mult=i / float(num_subchecks)):
                action *= (i - 1) / float(num_subchecks)
                break

        # Execute action
        self.agent_position = np.clip(self.agent_position + action, -1, 1)
        obs = self.obs()
        terminal = self.compute_terminal(self.goal, obs)
        reward = self.compute_reward(self.goal, obs)

        # check if max steps has been reached
        self.time_step += 1
        if self.time_step >= self.max_steps:
            terminal = True

        return obs, reward, terminal, {}

    def reset(self):
        self.agent_position = self.get_non_intersecting_position(self.agent_position_generator)
        self.goal = self.get_non_intersecting_position(self.goal_position_generator)
        self.time_step = 0
        return self.obs()

    def obs(self):
        return np.concatenate([self.agent_position, self.goal], axis=0)

    # Hindsight Stuff

    def at_goal(self, goal, obs):
        without_goal = obs[:-2]
        dist = np.linalg.norm(without_goal - goal, ord=2)
        return dist <= self.dist_cutoff

    def change_goal(self, goal, obs):
        without_goal = obs[:-2]
        return np.concatenate([without_goal, goal], axis=0)

    def compute_reward(self, goal, obs):
        return 1.0 if self.at_goal(goal, obs) else -0.01

    def compute_terminal(self, goal, obs):
        return self.at_goal(goal, obs)

    def obs_to_goal(self, obs):
        return obs[:2]

    # Rendering

    def render_agent(self):
        assert isinstance(self.agent_position, np.ndarray)
        assert isinstance(self.goal, np.ndarray)

        self.screen.fill((255, 255, 255))
        pos = (self.agent_position + 1) / 2 * self.image_size
        goal = (self.goal + 1) / 2 * self.image_size
        pygame.draw.circle(self.screen, (0, 0, 0), pos.astype(int), 3)
        pygame.draw.circle(self.screen, (255, 0, 0), goal.astype(int), 3)

        for obs in self.obstacles:
            obs.draw(self.screen, (0, 0, 0))
        if self.visualize:
            pygame.display.update()
            pygame.event.get()
        imgdata = pygame.surfarray.array3d(self.screen)
        imgdata.swapaxes(0, 1)
        return imgdata

    # Collision Handling

    def get_non_intersecting_position2(self, generator):
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

    def get_non_intersecting_position(self, generator):
        while True:
            position = generator()
            tl = self.image_size * (position + 1) / 2. - 0.5 * (5 / np.sqrt(2))
            agent_rect = pygame.Rect(tl[0], tl[1], 5 / np.sqrt(2), 5 / np.sqrt(2))
            assert len(position) == 2
            if not any(obstacle.collides(agent_rect)
                       for obstacle in self.obstacles):
                return position

    def check_intersects(self, agent_position, scaled_action, mult=1.0):
        position = np.clip(agent_position + mult * scaled_action, -1, 1)
        intersects = False

        tl = (self.image_size * (position + 1)) / 2. - 0.5 * 0.01
        agent_rect = pygame.Rect(tl[0], tl[1], 0.01, 0.01)
        for obstacle in self.obstacles:
            collision = obstacle.collides(agent_rect)
            # print(obstacle.rect.topleft, obstacle.rect.width, obstacle.rect.height)
            intersects |= collision
            if collision:
                obstacle.color = (255, 0, 0)
            else:
                obstacle.color = (0, 0, 0)
            if intersects:
                break
        return intersects


def instersects(position, obstacle):
    assert isinstance(position, np.ndarray)
    assert len(position) == 2
    assert isinstance(obstacle, pygame.rect.Rect)
    return obstacle.collidepoint(position)

    raise NotImplemented


class FourRoomExperiment(ContinuousGridworld):
    def __init__(self, visualize=False, image_size=64):
        from toy_environment import four_rooms_obstacle_list
        self.position_mapping = {0: [-0.75, -0.75], 1: [-0.75, 0.75], 2: [0.75, 0.75], 3: [0.75, -0.75]}
        super().__init__(four_rooms_obstacle_list.obstacle_list, visualize=visualize, image_size=image_size)

    def agent_position_generator(self):
        return np.array(self.position_mapping[np.random.randint(0, 4)])

    def goal_position_generator(self):
        return np.random.uniform(self.agent_position - 0.25, self.agent_position + .25)


def main():
    # env = FourRoomExperiment(visualize=True)
    env = ContinuousGridworld(room_obstacle_list.obstacle_list)
    # obs = env.reset()
    while True:
        action = {'s': [1.0, 0],
                  'w': [-1.0, 0],
                  'd': [0, 1.0],
                  'a': [0, -1.0]}.get(input('action:'), [0.0, 0.0])
        # action = np.random.uniform(-1, 1, size=2)
        obs, reward, terminal, info = env.step(action)
        # image = env.render_agent()
        if terminal:
            env.reset()


if __name__ == '__main__':
    main()
