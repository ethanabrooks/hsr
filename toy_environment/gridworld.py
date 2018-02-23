import numpy as np
import pygame
from gym import spaces
from pygame import rect

from environment.base import BaseEnv
from toy_environment import room_obstacle_list


def sample_position():
    return np.random.uniform(-1, 1, size=[2])


class Gridworld(BaseEnv):
    def __init__(self, obstacle_list_generator, visualize=False, image_size=64,
                 step_size=0.2, max_steps=1000, discrete=False):
        self.observation_space = spaces.Box(-1, 1, shape=[4])
        self.action_space = spaces.Box(-1, 1, shape=[2])
        self.image_size = image_size
        self.obstacles = obstacle_list_generator(image_size)

        self.agent_position = self.get_non_intersecting_position(self.agent_position_generator)
        self.goal = self.get_non_intersecting_position(self.goal_position_generator)
        if visualize:
            self.screen = pygame.display.set_mode((image_size, image_size), )
        else:
            self.screen = pygame.Surface((image_size, image_size))

        self.step_size = step_size
        self.geofence = 0.2
        self.max_steps = max_steps
        self.step = 0

        # required for OpenAI code
        self.metadata = {'render.modes': 'rgb_array'}
        self.reward_range = -np.inf, np.inf
        self.spec = None

    @staticmethod
    def seed(seed):
        np.random.seed(seed)

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

        # check for clipping
        num_subchecks = 4
        action /= float(num_subchecks)
        for _ in range(num_subchecks):
            candidate_position = np.clip(self.agent_position + action, -1, 1)
            intersecting = self.intersecting_obstacles(candidate_position)
            for obstacle in intersecting:
                obstacle.color = 255, 0, 0
            if intersecting:
                break  # candidate position is not valid
            self.agent_position = candidate_position

        # get other step return values
        obs = self.obs()
        terminal = self.compute_terminal(self.goal, obs)
        reward = self.compute_reward(self.goal, obs)

        # check if max steps has been reached
        self.step += 1
        if self.step >= self.max_steps:
            terminal = True

        return obs, reward, terminal, {}

    def reset(self):
        self.agent_position = self.get_non_intersecting_position(self.agent_position_generator)
        self.goal = self.get_non_intersecting_position(self.goal_position_generator)
        self.step = 0
        return self.obs()

    def obs(self):
        return np.concatenate([self.agent_position, self.goal], axis=0)

    # Hindsight Stuff

    def at_goal(self, goal, obs):
        without_goal = obs[:-2]
        dist = np.linalg.norm(without_goal - goal, ord=2)
        return dist <= self.geofence

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

    def render(self, mode='human'):
        assert isinstance(self.agent_position, np.ndarray)
        assert isinstance(self.goal, np.ndarray)

        self.screen.fill((255, 255, 255))
        pos = (self.agent_position + 1) / 2 * self.image_size
        goal = (self.goal + 1) / 2 * self.image_size
        pygame.draw.circle(self.screen, (0, 0, 0), pos.astype(int), 3)
        pygame.draw.circle(self.screen, (255, 0, 0), goal.astype(int), 3)

        for obs in self.obstacles:
            obs.draw(self.screen, (0, 0, 0))
        if mode == 'human':
            pygame.display.update()
            pygame.event.get()
        imgdata = pygame.surfarray.array3d(self.screen)
        imgdata.swapaxes(0, 1)
        return imgdata

    # Collision Handling

    def get_non_intersecting_position(self, generator):
        while True:
            position = generator()
            if not self.intersecting_obstacles(position):
                return position

    def intersecting_obstacles(self, agent_position):
        assert len(agent_position) == 2
        center = self.image_size * (agent_position + 1) / 2.
        offset = .1
        rect = pygame.Rect(*(center + offset), offset, offset)
        return [obstacle for obstacle in self.obstacles
                if obstacle.rect.colliderect(rect)]


class FourRoomExperiment(Gridworld):
    def __init__(self, visualize=False, image_size=64):
        from toy_environment import four_rooms_obstacle_list
        self.position_mapping = {0: [-0.75, -0.75],
                                 1: [-0.75, 0.75],
                                 2: [0.75, 0.75],
                                 3: [0.75, -0.75]}
        super().__init__(four_rooms_obstacle_list.obstacle_list,
                         image_size=image_size)

    def agent_position_generator(self):
        return np.array(self.position_mapping[np.random.randint(0, 4)])

    def goal_position_generator(self):
        return np.random.uniform(self.agent_position - 0.25, self.agent_position + .25)


def main():
    # env = FourRoomExperiment(visualize=True)
    pygame.init()
    env = Gridworld(room_obstacle_list.obstacle_list, visualize=True)
    actions = {pygame.K_d: [1, 0],
               pygame.K_a: [-1, 0],
               pygame.K_s: [0, 1],
               pygame.K_w: [0, -1]}
    # obs = env.reset()
    while True:
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        for key, pressed in enumerate(keys):
            if pressed:
                if key == pygame.K_SPACE:
                    print(env.agent_position)
                else:
                    action = np.array(actions.get(key, [0, 0])) * .1
                    env.step(action)
        env.render()


if __name__ == '__main__':
    main()
