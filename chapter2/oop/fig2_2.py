#!/usr/bin/env python3
"""Chapter 2.3, The 10-armed testbed,
action-value greedy method,
page 29"""

import time
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt

from bandit import Bandit, EpsGreedy


if __name__ == '__main__':

    # initialize
    runs = int(2e3)
    steps = int(1e3)
    args = [steps] * runs
    epsilons = (0, 0.1, 0.01)
    # comment this line if run on windows or OS X  (default method)
    mp.set_start_method('spawn')

    print('Stationary greedy started...')
    t1 = time.perf_counter()

    with mp.Pool(mp.cpu_count()) as pool:
        def func(x):
            return np.array(pool.map(EpsGreedy(eps=x).rews_opts_stat, args))
        result = [func(eps) for eps in epsilons]
        # get 3 (2000, 2, 1000)-shaped arrays, axis=1 stands for rewards and optimals

    t2 = time.perf_counter()
    print(f'Done in {round(t2 - t1, 3)} sec')

    # get the average rewards
    rewards = [pair[:, 0, :].mean(axis=0) for pair in result]
    # get the percentage of the optimal actions
    optimals = [Bandit.percent(pair[:, 1, :]) for pair in result]

    # plotting
    colors = ('green', 'blue', 'red')
    labels = (r'$\varepsilon=0$ (greedy)', r'$\varepsilon=0.1$', r'$\varepsilon=0.01$')

    Bandit.plot(rewards, labels, 'Average reward')
    Bandit.plot(optimals, labels, '% Optimal action')

    plt.show()
