"""Richard S. Sutton and Andrew G. Barto - Reinforcement Learning: An Introduction
Second edition, 2018

Chapter 2.10 (Summary), A parameter study of nonstationary case,
page 44
"""
import time
import random
from math import exp, log
import multiprocessing as mp
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from stationary_greedy import argmax, plot


def eps_greedy(steps, eps):
    """Nonstationary 10-armed bandit problem.
    Sample-average epsilon-greedy method.

    :param steps: Number of timesteps
    :type steps: int
    :param eps: The probability of choosing the exploration vs exploitation.
    :type eps: float
    :return: Rewards
    :rtype: list
    """
    q = list(np.random.normal(0, 1, size=10))           # true action values
    q_est = [0] * 10                                    # estimated action values
    action_counts = [0] * 10                            # action counter
    rewards = list()                                    # rewards on each step

    for i in range(steps):
        # choose an action
        if random.random() < 1 - eps:                   # exploitation
            action = argmax(q_est)
        else:                                           # exploration
            action = random.randint(0, 9)
        action_counts[action] += 1                      # update action counter
        rewards.append(np.random.normal(q[action], 1))  # get action normally distributed reward

        # update estimated values
        q_est[action] += (rewards[-1] - q_est[action]) / action_counts[action]

        # introduce true value fluctuations
        q += np.random.normal(0, 0.01, size=10)
    return rewards


def const_eps_greedy(steps, eps):
    """Nonstationary 10-armed bandit problem.
    Constant step-size (0.1) epsilon-greedy method.

    :param steps: Number of timesteps
    :type steps: int
    :param eps: The probability of choosing the exploration vs exploitation.
    :type eps: float
    :return: Rewards
    :rtype: list
    """
    q = list(np.random.normal(0, 1, size=10))           # true action values
    q_est = [0] * 10                                    # estimated action values
    rewards = list()                                    # rewards on each step

    for i in range(steps):
        # choose an action
        if random.random() < 1 - eps:                   # exploitation
            action = argmax(q_est)
        else:                                           # exploration
            action = random.randint(0, 9)
        rewards.append(np.random.normal(q[action], 1))  # get action normally distributed reward

        # update estimated values
        q_est[action] += (rewards[-1] - q_est[action]) * 0.1

        # introduce true value fluctuations
        q += np.random.normal(0, 0.01, size=10)
    return rewards


def grad_bline(steps, alpha):
    """Nonstationary 10-armed bandit problem.

    Gradient Bandit Algorithm with baseline.

    :param steps: Number of timesteps
    :type steps: int
    :param alpha: Constant step-size
    :type alpha: float
    :return: Rewards
    :rtype: list
    """
    q = list(np.random.normal(0, 1, size=10))                       # true action values
    h = [0] * 10                                                    # action preferences
    p = [0.1] * 10                                                  # probabilities of choosing an action
    mean = 0                                                        # mean reward initialisation
    rewards = list()

    for i in range(steps):
        action = random.choices(range(10), weights=p, k=1)[0]       # choose an action
        reward = np.random.normal(q[action], 1)                     # get action reward
        rewards.append(reward)
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

        # introduce true value fluctuations
        q += np.random.normal(0, 0.01, size=10)
    return rewards


def ucb(steps, c):
    """Stationary 10-armed bandit problem.
    Upper-Confidence-Bound Action Selection.

    :param steps: Number of time steps
    :type steps: int
    :param c: Degree of exploration
    :type c: float
    :return: Rewards
    :rtype: list
    """
    q = list(np.random.normal(0, 1, size=10))           # true action values
    q_est = [0] * 10                                    # estimated action values
    action_counts = [0] * 10                            # action counter
    ucb_q_est = [5] * 10                                # ucb estimations
    rewards = list()

    for i in range(0, steps):
        action = argmax(ucb_q_est)                       # choose greedy
        rewards.append(np.random.normal(q[action], 1))   # get action reward

        # update ucb estimations
        for j in range(10):
            if action_counts[j] != 0:
                sqrt = (log(i) / action_counts[j]) ** 0.5
                ucb_q_est[j] = q_est[j] + c * sqrt
        action_counts[action] += 1                       # update action counter

        # update estimated values
        q_est[action] += (rewards[-1] - q_est[action]) / action_counts[action]

        # introduce true value fluctuations
        q += np.random.normal(0, 0.01, size=10)
    return rewards


def greedy(steps, q0):
    """Stationary 10-armed bandit problem.
    Constant step-size greedy method

    :param steps: Number of timesteps
    :type steps: int
    :param q0: Initial value for q estimation
    :type q0: float
    :return: Rewards
    :rtype: list
    """
    q = list(np.random.normal(0, 1, size=10))                   # true action values
    q_est = [q0] * 10                                           # estimated action values
    rewards = list()                                            # rewards on each step

    for i in range(steps):
        action = argmax(q_est)                                  # choose action
        rewards.append(np.random.normal(q[action], 1))          # get action normally distributed reward
        q_est[action] += (rewards[-1] - q_est[action]) * 0.1    # update estimated values

        # introduce true value fluctuations
        q += np.random.normal(0, 0.01, size=10)
    return rewards


if __name__ == '__main__':

    runs = int(2e3)
    steps = int(2e5)

    # parameter values (powers of 2)
    params = [2 ** i for i in range(-7, 3)]
    # string representations for parameter values
    x_ticks = ['1/128', '1/64', '1/32', '1/16', '1/8', '1/4', '1/2', '1', '2', '4']
    # indices for the slices of parameter values
    param_slices = {'eps_greedy': (0, 6),
                    'const_eps_greedy': (0, 6),
                    'grad_bline': (2, 11),
                    'ucb': (3, 11),
                    'optimistic_greedy': (5, 11)}
    # dictionary to store obtained reward values for particular method
    rewards = defaultdict(list)

    # comment this line if run on windows or OS X  (default method)
    mp.set_start_method('spawn')

    # parallel execution
    with mp.Pool(16) as pool:
        t0 = time.perf_counter()
        for method, _slice in param_slices.items():

            print(f'{method} started with parameters:')
            t1 = time.perf_counter()

            (start, stop) = _slice
            for param, x in zip(params[start:stop], x_ticks[start:stop]):
                print(f'{x}', end=' ')
                # mean reward across all runs
                arr = np.array(pool.starmap(locals()[method], [(steps, param)] * runs)).mean(axis=0)
                # overall mean reward
                rewards[method].append(arr[100000:].mean())

            t2 = time.perf_counter()
            print(f'done in {round(t2 - t1, 3)} sec')

        t3 = time.perf_counter()
        print(f'Overall execution time {round(t3 - t0, 3)} sec')

    # plotting
    # labels and colors
    labels = (r'$\varepsilon$-greedy, $\varepsilon$',
              'constant step\n' r'$\varepsilon$-greedy $\alpha=0.1$, $\varepsilon$',
              r'gradient bandit, $\alpha$',
              r'UCB, $c$',
              'optimistic greedy\n' r'$\alpha=0.1, Q_0$')
    ylabel = 'Average reward over\n last 100 000 steps'
    xlabel = r'$\varepsilon, \alpha, c, Q_0$'
    colors = ('red', 'purple', 'green', 'blue', 'black')

    # x axis values to correspond with parameter slices
    x = [list(range(10)[start:stop]) for (start, stop) in param_slices.values()]
    # plots
    ax = plot(rewards.values(), labels, ylabel, datax=x, xlabel=xlabel, colors=colors, fig_size=(15, 8))
    plt.xticks(range(10), x_ticks)

    plt.show()
