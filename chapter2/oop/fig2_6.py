#!/usr/bin/env python3
"""Chapter 2.10 (Summary), A parameter study of nonstationary case,
page 44
"""
import time
import multiprocessing as mp
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from bandit import Bandit, EpsGreedy, GradientBaseline, UCB, EpsGreedyConstant


if __name__ == '__main__':

    runs = int(2e3)
    steps = int(1e3)
    args = [steps] * runs

    # parameter values (powers of 2)
    params = [2 ** i for i in range(-7, 3)]
    # string representations for parameter values
    x_ticks = ['1/128', '1/64', '1/32', '1/16', '1/8', '1/4', '1/2', '1', '2', '4']
    # indices for the slices of parameter values

    def methods(x):
        return {'eps_greedy': EpsGreedy(eps=x).rewards_stat,
                'grad_bline': GradientBaseline(alpha=x).rewards_stat,
                'ucb': UCB(c=x).rewards_stat,
                'optimistic_greedy': EpsGreedyConstant(eps=0, alpha=0.1, estim_value=x).rewards_stat}
    param_slices = {'eps_greedy': (0, 6),
                    'grad_bline': (2, 11),
                    'ucb': (3, 11),
                    'optimistic_greedy': (5, 11)}
    # dictionary to store obtained reward values for particular method
    rewards = defaultdict(list)

    # comment this line if run on windows or OS X  (default method)
    mp.set_start_method('spawn')

    # parallel execution
    with mp.Pool(mp.cpu_count()) as pool:
        t0 = time.perf_counter()
        for method, _slice in param_slices.items():

            print(f'{method} started with parameters:')
            t1 = time.perf_counter()

            (start, stop) = _slice
            for param, x in zip(params[start:stop], x_ticks[start:stop]):
                print(f'{x}', end=' ')
                # mean reward over all runs
                arr = np.array(pool.map(methods(param)[method], args)).mean(axis=0)
                # mean reward over all steps
                rewards[method].append(arr.mean())

            t2 = time.perf_counter()
            print(f'done in {round(t2 - t1, 3)} sec')

        t3 = time.perf_counter()
        print(f'Overall execution time {round(t3 - t0, 3)} sec')

    # plotting
    # labels and colors
    labels = (r'$\varepsilon$-optimistic_greedy, $\varepsilon$',
              r'gradient bandit, $\alpha$',
              r'UCB, $c$',
              'optimistic greedy\n' r'$\alpha=0.1, Q_0$')
    ylabel = 'Average reward over\n first 1000 steps'
    xlabel = r'$\varepsilon, \alpha, c, Q_0$'
    colors = ('red', 'green', 'blue', 'black')

    # x axis values to correspond with parameter slices
    x = [list(range(10)[start:stop]) for (start, stop) in param_slices.values()]
    # plots
    ax = Bandit.plot(rewards.values(), labels, ylabel, datax=x, xlabel=xlabel, colors=colors, fig_size=(15, 8))
    plt.xticks(range(10), x_ticks)

    plt.show()
