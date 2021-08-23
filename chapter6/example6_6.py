#!/usr/bin/env python3
"""Example 6.6, page 132"""

import copy
import time

import numpy as np
import matplotlib.pyplot as plt
from tqdm.contrib.concurrent import process_map
import multiprocessing as mp

from example6_5 import sarsa_step, eps_greedy_policy


class CliffWorld:

    def __init__(self, world_dimesions, actions, start_position, goal_position):

        # world limits
        self.width, self.height = world_dimesions[0] - 1, world_dimesions[1] - 1
        self.start = start_position
        self.goal = goal_position

        self.cliff = [(x, 0) for x in range(1, self.width)]

        # Actions: up, down, right, left
        self.actions = copy.copy(actions)

    def step(self, state, action):

        x, y = state

        # Change the coordinate by the selected action
        action_x, action_y = self.actions[action]
        x, y = x + action_x, y + action_y

        if (x, y) not in self.cliff:
            # Stay inside
            x = np.clip(x, 0, self.width)
            y = np.clip(y, 0, self.height)
            reward = -1

        else:
            x, y = copy.copy(self.start)
            reward = -100

        return (x, y), reward

    def render(self):
        width, height = self.width, self.height
        world = np.zeros((height + 1, width + 1))
        print(world)


def q_learning_step(world, q, state, action, alpha=0.5, eps=0.1):

    action = eps_greedy_policy(q, state, world.actions, eps=eps)
    next_state, reward = world.step(state, action)

    # State, action
    x0, y0, z0 = state[0], state[1], action

    # State'
    x1, y1, = next_state[0], next_state[1]

    # Action which return maximal Q
    z1 = np.argmax(q[x1, y1])

    q[x0, y0, z0] += alpha * (reward + q[x1, y1, z1] - q[x0, y0, z0])

    state = next_state

    return state, action, reward


def learning_cliff(args):

    method, world, alpha, episodes, eps = args
    state_action_dim = world.width + 1, world.height + 1, len(world.actions)
    q = np.zeros(state_action_dim)
    reward_sum_seq = list()

    for i in range(episodes):
        state = copy.copy(world.start)
        action = eps_greedy_policy(q, state, world.actions, eps=eps)
        rewards_sum = 0
        while state != world.goal:
            state, action, reward = method(world, q, state, action, alpha=alpha, eps=eps)
            rewards_sum += reward
        reward_sum_seq.append(rewards_sum)

    return reward_sum_seq


def method_results(world, method, runs, alpha=0.5, episodes=500, eps=0.1, chunksize=1):
    t0 = time.perf_counter()
    workers = mp.cpu_count()
    args = [(method, world, alpha, episodes, eps)] * runs
    result = np.array(process_map(learning_cliff, args, max_workers=workers, chunksize=chunksize))
    result = np.mean(result, axis=0)
    t1 = time.perf_counter()
    print(f'Done in {t1 - t0} sec')

    return result


def plot_example6_6(actions):

    dim = 12, 4
    start = 0, 0
    goal = 11, 0

    world = CliffWorld(dim, actions, start, goal)

    # To get smooth curves you have to average results over multiple runs
    runs = 100
    mp.set_start_method('spawn')

    print('Q-learning...')
    time.sleep(0.5)
    q_learning = method_results(world, q_learning_step, runs)
    plt.plot(q_learning, label='Q-learning')

    time.sleep(0.5)
    print('Sarsa...')
    time.sleep(0.5)
    sarsa = method_results(world, sarsa_step, runs)
    plt.plot(sarsa, label='Sarsa')

    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.ylim(-100, -20)
    plt.show()


if __name__ == '__main__':
    actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    plot_example6_6(actions)
