"""Richard S. Sutton and Andrew G. Barto - Reinforcement Learning: An Introduction
Second edition, 2018

Chapter 2.8, Gradient Bandit Algorithms,
page 35
"""
import time
import random
from math import exp
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt

from stationary_greedy import argmax, percent, plot


def grad_no_bline(steps, alpha):
    """Stationary 10-armed bandit problem.
    Gradient Bandit Algorithm without baseline.

    :param steps: Number of timesteps
    :type steps: int
    :param alpha: Constant step-size
    :type alpha: float
    :return: Optimals
    :rtype: list
    """
    q = list(np.random.normal(4, 1, size=10))                   # true action values
    h = [0] * 10                                                # action preferences
    p = [0.1] * 10                                              # probabilities of choosing an action
    optimals = list()                                           # list of bools
    optimal = argmax(q)

    for i in range(steps):
        action = random.choices(range(10), weights=p, k=1)[0]   # choose an action
        optimals.append(action is optimal)                      # check if the action had maximum value
        reward = np.random.normal(q[action], 1)                 # get action reward

        # update preferences
        h_exps = []
        for j, _ in enumerate(h):
            if j == action:
                h[j] = h[j] + alpha * reward * (1 - p[j])       # preference for chosen action
            else:
                h[j] = h[j] - alpha * reward * p[j]             # preference for other actions
            h_exps.append(exp(h[j]))                            # exponents for each preference

        # update action probabilities
        h_exps_sum = sum(h_exps)
        p = [x / h_exps_sum for x in h_exps]

    return optimals


def grad_bline(steps, alpha):
    """Stationary 10-armed bandit problem.
    Gradient Bandit Algorithm with baseline.

    :param steps: Number of timesteps
    :type steps: int
    :param alpha: Constant step-size
    :type alpha: float
    :return: Optimals
    :rtype: list
    """
    q = list(np.random.normal(4, 1, size=10))  # true action values
    h = [0] * 10  # action preferences
    p = [0.1] * 10  # probabilities of choosing an action
    mean = 0  # mean reward initialisation
    optimals = list()  # list of bools
    optimal = argmax(q)

    for i in range(steps):

        action = random.choices(range(10), weights=p, k=1)[0]       # choose an action
        optimals.append(action is optimal)                          # check if the action had maximum value
        reward = np.random.normal(q[action], 1)                     # get action reward

        # update preferences
        mean = mean + (reward - mean) / (i + 1)                     # incremental formula for mean value
        h_exps = []
        for j, _ in enumerate(h):
            if j == action:
                h[j] = h[j] + alpha * (reward - mean) * (1 - p[j])  # preference for chosen action
            else:
                h[j] = h[j] - alpha * (reward - mean) * p[j]        # preference for other actions
            h_exps.append(exp(h[j]))                                # exponents for each preference

        # update action probabilities
        h_exps_sum = sum(h_exps)
        p = [x / h_exps_sum for x in h_exps]

    return optimals


if __name__ == '__main__':

    runs = int(2e3)
    steps = int(1e3)

    # comment this line if run on windows or OS X  (default method)
    mp.set_start_method('spawn')

    print('Started gradient bandit...')
    t1 = time.perf_counter()

    with mp.Pool(mp.cpu_count()) as pool:
        bl01 = np.array(pool.starmap(grad_bline, [(steps, 0.1)] * runs))
        bl04 = np.array(pool.starmap(grad_bline, [(steps, 0.4)] * runs))
        no_bl01 = np.array(pool.starmap(grad_no_bline, [(steps, 0.1)] * runs))
        no_bl04 = np.array(pool.starmap(grad_no_bline, [(steps, 0.4)] * runs))

    t2 = time.perf_counter()
    print(f'Done in {round(t2 - t1, 3)} sec')

    result = [bl01, bl04, no_bl01, no_bl04]
    # get percentages
    result = [percent(i) for i in result]

    # plotting
    labels = (r'with baseline, $\alpha=0.1$',
              r'with baseline, $\alpha=0.4$',
              r'without baseline, $\alpha=0.1$',
              r'without baseline, $\alpha=0.4$')
    colors = ('blue', 'cornflowerblue', 'sienna', 'tan')

    plot(result, labels, '% Optimal action', colors=colors)
    plt.show()
