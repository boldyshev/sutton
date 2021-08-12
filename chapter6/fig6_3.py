#!/usr/bin/env python3
"""Figure 6.3, page 133"""

import copy
import time
import random
import pickle

import numpy as np
from numpy import argmax
import matplotlib.pyplot as plt
from tqdm.contrib.concurrent import process_map
import multiprocessing as mp

from example6_6 import CliffWorld, q_learning_step, learning_cliff, method_results
from example6_5 import sarsa_step, eps_greedy_policy

NON_OPTIMAL_PROBABILITY = 0.1 / 3
OPTIMAL_PROBABILITY = 0.9

def sarsa_expected_step(world, q, state, action, alpha=0.5, eps=0.1):

    next_state, reward = world.step(state, action)

    # State, action
    x, y, z = state[0], state[1], action
    x1, y1 = next_state
    next_action = argmax(q[x1, y1])

    q_expected = 0
    for _z in range(4):
        incr = NON_OPTIMAL_PROBABILITY * q[x1, y1, _z]
        if _z == next_action:
            incr = OPTIMAL_PROBABILITY * q[x1, y1, _z]
        q_expected += incr

    q[x, y, z] += alpha * (reward + q_expected - q[x, y, z])

    if random.random() < eps:
        next_action = random.randint(0, 3)

    state, action = next_state, next_action

    return state, action, reward


def get_averages(world, method, runs, episodes, name):
    print(name)
    result = list()
    for alpha in np.linspace(0.1, 1, num=10):
        print(f'  alpha = {alpha}')
        time.sleep(0.05)
        avg = method_results(world, method, runs, alpha=alpha, episodes=episodes, chunksize=1).mean()
        result.append(avg)
        print('result = ', avg)

    with open(f'{name}.pickle', 'wb') as f:
        pickle.dump(result, f)

    return result


if __name__ == '__main__':

    dim = 12, 4
    start = 0, 0
    goal = 11, 0
    actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    mp.set_start_method('spawn')

    world = CliffWorld(dim, actions, start, goal)

    # Interim performance
    runs = 50_000
    episodes = 100
    names = 'q-learning', 'sarsa', 'expected-sarsa'
    methods = q_learning_step, sarsa_step, sarsa_expected_step
    for name, method in zip(names, methods):
        get_averages(world, method, runs, episodes, 'inter-' + name)

    # Asymptotic performance
    runs = 10
    episodes = 100_000
    names = 'q-learning', 'sarsa', 'expected-sarsa'
    methods = q_learning_step, sarsa_step, sarsa_expected_step
    for name, method in zip(names, methods):
        get_averages(world, method, runs, episodes, 'asympt-' + name)
