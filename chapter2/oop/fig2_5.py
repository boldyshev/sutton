"""Richard S. Sutton and Andrew G. Barto - Reinforcement Learning: An Introduction
Second edition, 2018

Chapter 2.8, Gradient Bandit Algorithms,
page 35
"""
import time
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt

from bandit import Bandit, GradientNoBaseline, GradientBaseline

if __name__ == '__main__':

    runs = int(2e3)
    steps = int(1e3)
    args = [steps] * runs

    # comment this line if run on windows or OS X  (default method)
    mp.set_start_method('spawn')

    print('Started gradient bandit...')
    t1 = time.perf_counter()

    with mp.Pool(mp.cpu_count()) as pool:
        bl01 = np.array(pool.map(GradientBaseline(true_value=4, alpha=0.1).optimals_stat, args))
        bl04 = np.array(pool.map(GradientBaseline(true_value=4, alpha=0.4).optimals_stat, args))
        no_bl01 = np.array(pool.map(GradientNoBaseline(true_value=4, alpha=0.1).optimals_stat, args))
        no_bl04 = np.array(pool.map(GradientNoBaseline(true_value=4, alpha=0.4).optimals_stat, args))

    t2 = time.perf_counter()
    print(f'Done in {round(t2 - t1, 3)} sec')

    result = [bl01, bl04, no_bl01, no_bl04]
    # get percentages
    result = [Bandit.percent(i) for i in result]

    # plotting
    labels = (r'with baseline, $\alpha=0.1$',
              r'with baseline, $\alpha=0.4$',
              r'without baseline, $\alpha=0.1$',
              r'without baseline, $\alpha=0.4$')
    colors = ('blue', 'cornflowerblue', 'sienna', 'tan')

    Bandit.plot(result, labels, '% Optimal action', colors=colors)
    plt.show()
