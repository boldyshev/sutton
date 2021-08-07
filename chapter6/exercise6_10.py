#!/usr/bin/env python3
"""Exercise 6.9, page 131"""


import numpy as np
import matplotlib.pyplot as plt

from example6_5 import WindyGridworld
from exercise6_9 import plot_time_episodes


class StochasticWind(WindyGridworld):

    def step(self, state, action):
        """Version with stochastic wind"""

        stochastic_wind = np.random.choice([-1, 0, 1])

        x, y = state
        y += self.wind[x] + stochastic_wind
        action_x, action_y = self.actions[action]
        x = np.clip(x + action_x, 0, self.max_x)
        y = np.clip(y + action_y, 0, self.max_y)
        reward = -1

        return (x, y), reward


if __name__ == '__main__':
    actions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)]
    plot_time_episodes(StochasticWind, actions)

    actions_halt = actions + [(0, 0)]
    plot_time_episodes(StochasticWind, actions_halt)

    plt.xlabel('Time steps')
    plt.ylabel('Episodes')
    plt.legend()
    plt.show()
