#!/usr/bin/env python3
"""Figure 6.3, page 133"""

import time
import random
import pickle

import numpy as np
from numpy import argmax
import matplotlib.pyplot as plt
import multiprocessing as mp

from example6_6 import CliffWorld, q_learning_step, method_results
from example6_5 import sarsa_step

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


def plot_results():
    dim = 12, 4
    start = 0, 0
    goal = 11, 0
    actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    world = CliffWorld(dim, actions, start, goal)

    mp.set_start_method('spawn')

    names = 'q-learning', 'sarsa', 'expected-sarsa'
    methods = q_learning_step, sarsa_step, sarsa_expected_step

    # Interim performance
    runs = 50_000
    episodes = 100
    for name, method in zip(names, methods):
        name = 'inter-' + name
        averages = get_averages(world, method, runs, episodes, name)
        plt.plot(averages, label=name)

    # Asymptotic performance
    runs = 10
    episodes = 100_000
    for name, method in zip(names, methods):
        name = 'asympt-' + name
        averages = get_averages(world, method, runs, episodes, name)
        plt.plot(averages, label=name)

    plt.legend()
    plt.ylim(-140, 0)
    plt.show()


def plot_saved_results():
    names = 'q-learning', 'sarsa', 'expected-sarsa'
    inter = dict()
    asymp = dict()
    for name in names:
        with open(f'inter-{name}.pickle', 'rb') as f:
            inter[name] = pickle.load(f)

        with open(f'asympt-{name}.pickle', 'rb') as f:
            asymp[name] = pickle.load(f)

    for name in names:
        y = inter[name]
        y1 = asymp[name]
        plt.plot(y, label='inter-' + name)
        plt.plot(y1, label='asymp-' + name)

    plt.legend()
    plt.ylim(-140, 0)
    plt.show()


if __name__ == '__main__':
    plot_saved_results()
