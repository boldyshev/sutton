#!/usr/bin/env python3
"""Chapter 2.3, The 10-armed tesbed,
action-value epsilon-greedy methods,
page 29
"""
import time
import random
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def argmax(iterable):
    """Returns the index of the first maximum element for python built-in iterables (e.g. lists or tuples).
    Turns out to be faster than numpy.argmax on low-dimensional vectors.

    :param iterable: Input iterable
    :type iterable: iterable
    :return: Maximum element index
    :rtype: int
    """

    return max(range(len(iterable)), key=lambda x: iterable[x])


def greedy(steps):
    """Stationary 10-armed bandit problem.
    Sample-average greedy method.

    :param steps: Number of timesteps
    :type steps: int
    :return: Two lists: rewards and if the chosen action was optimal
    :rtype: tuple
    """

    q = np.random.normal(0, 1, size=10)           # true action values
    q_est = [0] * 10                                    # estimated action values
    action_counts = [0] * 10                            # action counter
    rewards = list()                                    # rewards on each step
    optimals = list()                                   # bool array: 1 - max action value, otherwise 0
    optimal = argmax(q)
    for i in range(steps):
        action = argmax(q_est)                          # choose action
        action_counts[action] += 1                      # update action counter
        optimals.append(action is optimal)              # check if the action had maximum value
        rewards.append(np.random.normal(q[action], 1))  # get action normally distributed reward
        # update estimated values
        q_est[action] += (rewards[-1] - q_est[action]) / action_counts[action]

    return rewards, optimals


def eps_greedy(steps, eps):
    """Stationary 10-armed bandit problem.
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
    optimal = argmax(q)
    for i in range(steps):
        if random.random() < 1 - eps:                   # exploitation
            action = argmax(q_est)                      # chose max of the estimated values
        else:                                           # exploration
            action = random.randint(0, 9)               # choose random action
        action_counts[action] += 1                      # update action counter
        optimals.append(action is optimal)              # check if the action had maximum value
        rewards.append(np.random.normal(q[action], 1))  # get action normally distributed reward
        # update estimated values
        q_est[action] += (rewards[-1] - q_est[action]) / action_counts[action]

    return rewards, optimals


def percent(arr):
    """Optimal actions percentage

    :param arr: Optimal actions binary vector
    :type arr: numpy.array
    :return: Optimal action percentage in arr
    :rtype: float
    """

    return 100 * np.sum(arr, axis=0) / arr.shape[0]


def plot(datay,
         labels,
         ylabel,
         datax=None,
         xlabel='Steps',
         colors=('blue', 'red', 'green', 'black'),
         fig_size=(18, 8),
         font_size=20):
    """Plotting average rewards or optimal actions

    :param datay: The set of Y-axis values to be plotted in one graph
    :type datay: iterable
    :param labels: Legend labels
    :type labels: iterable
    :param ylabel: Y-axis label
    :type ylabel: str
    :param datax: X-axis values
    :type datax: iterable
    :param xlabel: X-axis label
    :type xlabel: str
    :param colors: Plot colors
    :type colors: iterable
    :param fig_size: Figure size
    :type fig_size: tuple
    :param font_size: Font size
    :type font_size: int
    :return: Matplotlib axes object
    :rtype: matplotlib.axes
    """
    # create figure
    fig, ax = plt.subplots(figsize=fig_size)

    # plot graphs
    for i, arr in enumerate(datay):
        if not datax:
            x = range(len(arr))
        else:
            x = datax[i]
        ax.plot(x, arr, label=labels[i], color=colors[i])

    # labels etc.
    ax.set_ylabel(ylabel, fontsize=font_size)
    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.tick_params(axis="x", labelsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend(loc='lower right', fontsize=font_size)
    if ylabel == '% Optimal action':
        ax.yaxis.set_major_formatter(PercentFormatter())
        ax.set_ylim(0, 100)
    plt.rc('mathtext', fontset="cm")
    return ax


if __name__ == '__main__':

    runs = int(2e3)
    steps = int(1e3)

    # comment this line if run on windows or OS X  (default method)
    mp.set_start_method('spawn')

    print('Stationary greedy started...')
    t1 = time.perf_counter()

    with mp.Pool(mp.cpu_count()) as pool:
        # 3 epsilons, 2000 runs
        eps01 = np.array(pool.starmap(eps_greedy, [(steps, 0.1)] * runs))
        eps001 = np.array(pool.starmap(eps_greedy, [(steps, 0.01)] * runs))
        greedy = np.array(pool.starmap(greedy, [(steps,)] * runs))
        # got (2000, 2, 1000)-shaped arrays, axis=1 stands for rewards and optimals

    t2 = time.perf_counter()
    print(f'Done in {round(t2 - t1, 3)} sec')

    # transpose the arrays to shape (2, 2000, 1000)
    result = eps01, eps001, greedy
    # result = [np.transpose(i, (1, 0, 2)) for i in result]

    # get the average rewards
    rewards = [pair[:, 0, :].mean(axis=0) for pair in result]
    # get the percentage of the optimal actions
    optimals = [percent(pair[:, 1, :]) for pair in result]

    # plotting
    labels = (r'$\varepsilon=0.1$',
              r'$\varepsilon=0.01$',
              'greedy')
    plot(rewards, labels, 'Average reward')
    plot(optimals, labels, '% Optimal action')

    plt.show()
