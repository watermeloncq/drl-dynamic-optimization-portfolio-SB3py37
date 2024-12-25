import gym.wrappers
import numpy as np
from ..config import eps
# from ..util import relu


class MinmaxActions(gym.Wrapper):
    """
    Environment wrapper to tanh actions.

    Usage:
        env = gym.make('Pong-v0')
        env = TanhActions(env)

    Ref: https://github.com/openai/gym/blob/master/gym/wrappers/README.md

    """

    def step(self, action):
        # also it puts it in a list
        if isinstance(action, list):
            action = action[0]

        if isinstance(action, dict):
            action = list(action[k] for k in sorted(action.keys()))

        action = (action - action.min()) / (action.max() - action.min() + eps)
        # action = (action - 0.5) * 2
        # action[0] = np.clip(action[0], 0, 1)

        # # arbitrage setting
        # n = np.size(action)
        # x1 = x2 = 0
        # for i in range(1,n):
        #     x1 += action[i]
        #     x2 += np.abs(action[i])
        # if x1 == x2 and x2-np.abs(action[-1]) != 0 :
        #     action[-1] = -action[-1]

        # scale action
        # action_abs = np.abs(action)
        # action /= action_abs.sum()
        # action /= 1

        return self.env.step(action)
