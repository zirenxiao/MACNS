from abc import abstractmethod

import gym


class AbstractEnv(gym.Env):
    @abstractmethod
    def action_masks(self):
        pass
