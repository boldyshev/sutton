#!/usr/bin/env python3
"""Chapter 2.6, The 10-armed tesbed, the reproduction of the experiment,
stationary, exponential recency-weighted average method (constant step-size),
optimistic initial values,
page 34
"""
import time
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt

from bandit import Bandit, EpsGreedyConstant


if __name__ == '__main__':

    steps = int(1e3)
    runs = int(2e3)
    args = [steps] * runs

    # comment this line if run on windows or OS X (default method)
    mp.set_start_method('spawn')

    print('Optimistic vs realistic started...')
    t1 = time.perf_counter()

    with mp.Pool(mp.cpu_count()) as pool:
        realistic = (EpsGreedyConstant(eps=0.1, alpha=0.1).optimals_stat, args)
        realistic = np.array(pool.map(*realistic))
        optimistic = (EpsGreedyConstant(estim_value=5, eps=0, alpha=0.1).optimals_stat, args)
        optimistic = np.array(pool.map(*optimistic))

    t2 = time.perf_counter()
    print(f'Done in {round(t2 - t1, 3)} sec')

    # percentage of optimal actions
    realistic = Bandit.percent(realistic)
    optimistic = Bandit.percent(optimistic)

    # plotting
    labels = ('Realistic, greedy\n' r'$Q_1=0, \varepsilon=0$',
              r'Optimistic, $\varepsilon$-greedy' '\n' r'$Q_1=5, \varepsilon=0.1$')

    Bandit.plot((realistic, optimistic), labels, '% Optimal action', colors=('grey', 'dodgerblue'))

    plt.show()
