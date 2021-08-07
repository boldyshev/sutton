#!/usr/bin/env python3
"""Exercise 6.5, page 130"""

import numpy as np
import matplotlib.pyplot as plt

from example6_5 import WindyGridworld, sarsa_windy


def exercise6_9():
    dim = 10, 7
    start = 0, 3
    goal = 7, 3
    actions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)]
    state_action_dim = *dim, len(actions)

    world = WindyGridworld(dim, actions, start, goal)
    q = np.zeros(state_action_dim)
    x, y = sarsa_windy(world, q)

    plt.plot(x, y)
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')
    plt.show()


if __name__ == '__main__':
    exercise6_9()
