#!/usr/bin/env python3
"""Example 6.5, page 130"""

import random
import copy

import numpy as np
from numpy import argmax
import matplotlib.pyplot as plt


class WindyGridworld:

    def __init__(self, world_dimesions, actions, start_position, goal_position):
        # world limits
        self.max_x, self.max_y = world_dimesions[0] - 1, world_dimesions[1] - 1
        self.start = start_position
        self.goal = goal_position

        self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

        # Actions: up, down, right, left
        self.actions = copy.copy(actions)

    def step(self, state, action):

        x, y = state

        # Wind impact
        y += self.wind[x]

        # Change the coordinate by the selected action
        action_x, action_y = self.actions[action]

        # Stay inside
        x = np.clip(x + action_x, 0, self.max_x)
        y = np.clip(y + action_y, 0, self.max_y)
        reward = -1

        return (x, y), reward


def eps_greedy_policy(q, state, actions, eps=0.1):

    # exploitation
    if random.random() > eps:
        action = argmax(q[state[0], state[1]])
    # exploration
    else:
        action = random.randint(0, len(actions) - 1)

    return action


def sarsa_step(world, q, state, action, alpha=0.5):

    next_state, reward = world.step(state, action)
    next_action = eps_greedy_policy(q, next_state, world.actions)

    # State, action
    x0, y0, z0 = state[0], state[1], action

    # State', action'
    x1, y1, z1 = next_state[0], next_state[1], next_action
    q[x0, y0, z0] += alpha * (reward + q[x1, y1, z1] - q[x0, y0, z0])

    state, action = next_state, next_action

    return state, action


def sarsa_windy(world, q, alpha=0.5, timesteps=8000):
    episode_counter = 0
    step_counter = 0
    timestep_seq = list()
    episodes_seq = list()

    while step_counter < timesteps:
        state = copy.copy(world.start)
        action = eps_greedy_policy(q, state, world.actions)

        while state != world.goal:
            state, action = sarsa_step(world, q, state, action, alpha=alpha)

            step_counter += 1
            timestep_seq.append(step_counter)
            episodes_seq.append(episode_counter)

        episode_counter += 1

    return timestep_seq, episodes_seq


def solve_world(actions, world_type):
    dim = 10, 7
    start = 0, 3
    goal = 7, 3
    state_action_dim = *dim, len(actions)

    world = world_type(dim, actions, start, goal)
    q = np.zeros(state_action_dim)
    x, y = sarsa_windy(world, q)

    plt.plot(x, y)
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')
    plt.show()


if __name__ == '__main__':
    actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    solve_world(actions, WindyGridworld)
