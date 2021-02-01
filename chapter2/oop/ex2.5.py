"""Richard S. Sutton and Andrew G. Barto - Reinforcement Learning: An Introduction
Second edition, 2018

Chapter 2.5, exercise 2.5,
page 33
"""
import time
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt

from bandit import Bandit, EpsGreedy, EpsGreedyConstant


if __name__ == '__main__':

    runs = int(2e3)
    steps = int(1e4)
    args = [steps] * runs

    # comment this line if run on windows or OS X (default method)
    mp.set_start_method('spawn')

    print('Start exercise 2.5... ')
    t1 = time.perf_counter()

    with mp.Pool(mp.cpu_count()) as pool:
        sample_av = np.array(pool.map(EpsGreedy(eps=0.1).rews_opts_nonstat, args))
        const_step = np.array(pool.map(EpsGreedyConstant(eps=0.1, alpha=0.1).rews_opts_nonstat, args))
        # got (2000, 2, 1000)-shaped arrays, axis=1 stands for rewards and optimals

    t2 = time.perf_counter()
    print(f'Done in {round(t2 - t1, 3)} sec')

    # get average rewards
    rewards = (sample_av[:, 0, :].mean(axis=0),
               const_step[:, 0, :].mean(axis=0))

    # get optimal action percentage
    optimals = (Bandit.percent(sample_av[:, 1, :]),
                Bandit.percent(const_step[:, 1, :]))

    # plot
    labels = ('Sample average\n' r'$\varepsilon=0.1$',
              'Constant step-size\n' r'$\varepsilon=0.1, \alpha=0.1$')
    Bandit.plot(rewards, labels, 'Average reward')
    Bandit.plot(optimals, labels, '% Optimal action')

    plt.show()
