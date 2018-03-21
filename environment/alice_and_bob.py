from environment.base import BaseEnv


class AliceEnv(BaseEnv):
    def _compute_reward(self, goal, obs, action):
        if action[0] > .5:
            self.end_state = self._obs()
            bobs_reward = self.bob.play()
            return -bobs_reward
        return 0


class Bob(BaseEnv):
    def __init__(self, max_steps, history_len, image_dimensions, neg_reward, steps_per_action):
        super().__init__(max_steps, history_len, image_dimensions, neg_reward, steps_per_action)
        self.__goal = None

    def _set_new_goal(self):
        alice_end_state = self.alice.play()
        self.__goal = alice_end_state

    def _goal(self):
        return self.__goal
