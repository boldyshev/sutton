#!/usr/bin/env python3
"""Exercise 6.9, page 130"""

import numpy as np
import matplotlib.pyplot as plt

from example6_5 import WindyGridworld, sarsa_windy


def plot_time_episodes(world_type, actions):
    dim = 10, 7
    start = 0, 3
    goal = 7, 3
    state_action_dim = *dim, len(actions)

    world = world_type(dim, actions, start, goal)
    q = np.zeros(state_action_dim)
    x, y = sarsa_windy(world, q)

    plt.plot(x, y, label=f'# actions {len(actions)}')


if __name__ == '__main__':
    actions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)]
    plot_time_episodes(WindyGridworld, actions)

    actions_halt = actions + [(0, 0)]
    plot_time_episodes(WindyGridworld, actions_halt)

    plt.xlabel('Time steps')
    plt.ylabel('Episodes')
    plt.legend()
    plt.show()
