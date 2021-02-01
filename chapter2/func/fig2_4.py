"""Richard S. Sutton and Andrew G. Barto - Reinforcement Learning: An Introduction
Second edition, 2018

Chapter 2.7, Upper-Confidence-Bound Action Selection,
page 35
"""
import time
import random
import multiprocessing as mp
from math import log

import numpy as np
import matplotlib.pyplot as plt

from fig2_2 import argmax, plot


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

        # update ucb estimations
        for j in range(10):
            if action_counts[j] != 0:
                sqrt = (log(i) / action_counts[j]) ** 0.5
                ucb_q_est[j] = q_est[j] + c * sqrt
        action_counts[action] += 1                       # update action counter
        rewards.append(np.random.normal(q[action], 1))   # get action reward

        # update estimated values
        q_est[action] += (rewards[-1] - q_est[action]) / action_counts[action]

    return rewards


def eps_greedy(steps, eps):
    """Stationary 10-armed bandit problem.
    Sample-average epsilon-greedy method.

    :param steps: Number of timesteps
    :type steps: int
    :param eps: The probability of choosing the exploration vs exploitation
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

    return rewards


if __name__ == '__main__':

    runs = int(2e3)      # the number of different bandit experiments
    steps = int(1e3)     # number of learning iterations in a single experiment

    # comment this line if run on windows or OS X  (default method)
    mp.set_start_method('spawn')

    print('Start upper confidence bound...')
    t1 = time.perf_counter()

    with mp.Pool(mp.cpu_count()) as pool:
        rewards_ucb = np.array(pool.starmap(ucb, [(steps, 2)] * runs))
        rewards_greedy = np.array(pool.starmap(eps_greedy, [(steps, 0.1)] * runs))

    t2 = time.perf_counter()
    print(f'Done in {round(t2 - t1, 3)} sec')

    # get the averages
    rewards_ucb = rewards_ucb.mean(axis=0)
    rewards_greedy = rewards_greedy.mean(axis=0)

    # plot
    labels = (r'UCB, $c=2$',
              r'$\varepsilon$-greedy, $\varepsilon=0.1$')
    plot((rewards_ucb, rewards_greedy), labels, 'Average reward', colors=('blue', 'grey'))

    plt.show()
