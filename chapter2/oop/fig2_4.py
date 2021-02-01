"""Richard S. Sutton and Andrew G. Barto - Reinforcement Learning: An Introduction
Second edition, 2018

Chapter 2.7, Upper-Confidence-Bound Action Selection,
page 35
"""
import time
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt

from bandit import Bandit, UCB, EpsGreedy

if __name__ == '__main__':

    runs = int(2e3)      # the number of different bandit experiments
    steps = int(1e3)     # number of learning iterations in a single experiment
    args = [steps] * runs

    # comment this line if run on windows or OS X  (default method)
    mp.set_start_method('spawn')

    print('Start upper confidence bound...')
    t1 = time.perf_counter()

    with mp.Pool(mp.cpu_count()) as pool:
        ucb = np.array(pool.map(UCB(c=2).stationary, args))[:, 0, :]
        greedy = np.array(pool.map(EpsGreedy(eps=0.1).stationary, args))[:, 0, :]

    t2 = time.perf_counter()
    print(f'Done in {round(t2 - t1, 3)} sec')

    # get the averages
    ucb = ucb.mean(axis=0)
    greedy = greedy.mean(axis=0)

    # plot
    labels = (r'UCB, $c=2$',
              r'$\varepsilon$-greedy, $\varepsilon=0.1$')
    Bandit.plot((ucb, greedy), labels, 'Average reward', colors=('blue', 'grey'))

    plt.show()
