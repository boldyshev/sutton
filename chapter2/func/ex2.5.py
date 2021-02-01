#!/usr/bin/env python3
"""Richard S. Sutton and Andrew G. Barto - Reinforcement Learning: An Introduction
Second edition, 2018

Chapter 2.5, exercise 2.5,
page 33
"""
import time
import random
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt

from stationary_greedy import argmax, percent, plot


def sample_average(steps, eps):
    """Nonstationary 10-armed bandit problem.
    Sample-average epsilon-greedy method.

    :param steps: Number of timesteps
    :type steps: int
    :param eps: The probability of choosing the exploration vs exploitation.
    :type eps: float
    :return: Two lists: rewards and if the chosen action was optimal
    :rtype: tuple
    """

    q = list(np.random.normal(0, 1, size=10))           # true action values
    q_est = [0] * 10                                    # estimated action values
    action_counts = [0] * 10                            # action counter
    rewards = list()                                    # rewards on each step
    optimals = list()                                   # bool array: 1 - max action value, otherwise 0

    for i in range(steps):
        # choose action
        if random.random() < 1 - eps:
            action = argmax(q_est)                      # exploitation
        else:
            action = random.randint(0, 9)               # exploration
        action_counts[action] += 1                      # update action counter
        optimals.append(action == argmax(q))            # check if the action had maximum value
        rewards.append(np.random.normal(q[action], 1))  # get a normally distributed reward
        # update estimated values
        q_est[action] += (rewards[-1] - q_est[action]) / action_counts[action]

        # introduce some random value fluctuations
        q += np.random.normal(0, 0.01, size=10)

    return rewards, optimals


def constant_step(steps, eps, alpha):

    """Nonstationary 10-armed bandit problem.
    Exponential recency-weighted average method (constant step-size).

    :param steps: Number of timesteps
    :type steps: int
    :param eps: The probability of choosing the exploration vs exploitation.
    :type eps: float
    :param alpha: Constant step-size
    :type alpha: float
    :return: Two lists: rewards and if the chosen action was optimal
    :rtype: tuple
    """
    q = list(np.random.normal(0, 1, size=10))           # true action values
    q_est = [0] * 10                                    # estimated action values
    rewards = list()                                    # rewards on each step
    optimals = list()                                   # bool array: 1 - max action value, otherwise 0

    for i in range(steps):
        # choose action
        if random.random() < 1 - eps:
            action = argmax(q_est)                      # exploitation
        else:
            action = random.randint(0, 9)               # exploration
        optimals.append(action == argmax(q))            # check if the action had maximum value
        rewards.append(np.random.normal(q[action], 1))  # get a normally distributed reward
        # update estimated values
        q_est[action] += (rewards[-1] - q_est[action]) * alpha

        # introduce true value fluctuations
        q += np.random.normal(0, 0.01, size=10)

    return rewards, optimals


if __name__ == '__main__':

    runs = int(2e3)
    steps = int(1e4)

    # comment this line if run on windows or OS X (default method)
    mp.set_start_method('spawn')

    print('Start exercise 2.5... ')
    t1 = time.perf_counter()

    with mp.Pool(mp.cpu_count()) as pool:
        sample_av = np.array(pool.starmap(sample_average, [(steps, 0.1)] * runs))
        const_step = np.array(pool.starmap(constant_step, [(steps, 0.1, 0.1)] * runs))
        # got (2000, 2, 1000)-shaped arrays, axis=1 stands for rewards and optimals

    t2 = time.perf_counter()
    print(f'Done in {round(t2 - t1, 3)} sec')

    # reshape the arrays to distinguish rewards and optimal actions
    sample_av = np.transpose(sample_av, (1, 0, 2))
    const_step = np.transpose(const_step, (1, 0, 2))

    # get average rewards
    rewards = (sample_av[0].mean(axis=0),
               const_step[0].mean(axis=0))

    # get optimal action percentage
    optimals = (percent(sample_av[1]),
                percent(const_step[1]))

    # plot
    labels = ('Sample average\n' r'$\varepsilon=0.1$',
              'Constant step-size\n' r'$\varepsilon=0.1, \alpha=0.1$')
    plot(rewards, labels, 'Average reward')
    plot(optimals, labels, '% Optimal action')

    plt.show()
