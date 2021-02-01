"""Richard S. Sutton and Andrew G. Barto - Reinforcement Learning: An Introduction
Second edition, 2018

Chapter 2.6, The 10-armed tesbed, the reproduction of the experiment,
stationary, exponential recency-weighted average method (constant step-size),
optimistic initial values,
page 34
"""
import time
import random
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt

from stationary_greedy import argmax, percent, plot


def realistic(steps, eps, alpha):
    """Realistic constant step-size epsilon-greedy method.

    :param steps: Number of timesteps
    :type steps: int
    :param eps: The probability of choosing the exploration vs exploitation.
    :type eps: float
    :param alpha: Constant step-size
    :type alpha: float
    :return: Two lists: rewards and if the chosen action was optimal
    :rtype: tuple
    """
    q = list(np.random.normal(0, 1, size=10))               # true action values
    q_est = [0] * 10                                        # estimated action values
    optimals = list()
    optimal = argmax(q)
    for i in range(steps):
        # choose action
        if random.random() < 1 - eps:
            action = argmax(q_est)                          # exploitation
        else:
            action = random.randint(0, 9)                   # exploration
        optimals.append(action is optimal)
        reward = np.random.normal(q[action], 1)             # get a reward
        q_est[action] += (reward - q_est[action]) * alpha   # update estimated values (alpha = 0.1)

    return optimals


def optimistic(steps, alpha):
    """Optimistic constant step-size greedy method.

    :param steps: Number of timesteps
    :type steps: int
    :param alpha: Constant step-size
    :type alpha: float
    :return: Two lists: rewards and if the chosen action was optimal
    :rtype: tuple
    """
    q = list(np.random.normal(0, 1, size=10))       # true action values
    q_est = [5] * 10                                # estimated action values are +5 to true values (optimistic)
    optimals = list()
    optimal = argmax(q)
    for i in range(steps):
        a = argmax(q_est)                           # greedy approach
        optimals.append(a is optimal)               # check if the action had maximum value
        reward = np.random.normal(q[a], 1)          # get a reward
        q_est[a] += (reward - q_est[a]) * alpha     # update estimated values (alpha = 0.1)

    return optimals


if __name__ == '__main__':

    steps = int(1e3)
    runs = int(2e3)

    # comment this line if run on windows or OS X (default method)
    mp.set_start_method('spawn')

    print('Optimistic vs realistic started...')
    t1 = time.perf_counter()

    with mp.Pool(mp.cpu_count()) as pool:
        real = np.array(pool.starmap(realistic, [(steps, 0.1, 0.1)] * runs))
        opt = np.array(pool.starmap(optimistic, [(steps, 0.1)] * runs))

    t2 = time.perf_counter()
    print(f'Done in {round(t2 - t1, 3)} sec')

    # percentage of optimal actions
    real = percent(real)
    opt = percent(opt)

    # plotting
    labels = ('Realistic, greedy\n' r'$Q_1=0, \varepsilon=0$',
              r'Optimistic, $\varepsilon$-greedy' '\n' r'$Q_1=5, \varepsilon=0.1$')

    plot((real, opt), labels, '% Optimal action', colors=('grey', 'dodgerblue'))

    plt.show()
