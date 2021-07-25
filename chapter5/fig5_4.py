#!/usr/bin/env python3

"""Figure 5.3, page 107"""

import time
from random import choices, getrandbits
import multiprocessing as mp

from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np


def generate_episode():
    """One episode"""

    steps_counter = 1

    while True:
        # 1 for left, 0 for right
        action = getrandbits(1)

        # If the action doesn't match the target policy, importance ratio is 0 so the return is also 0
        if not action:
            return 0

        terminal = choices((True, False), weights=(0.1, 0.9))[0]

        # The numerator of the importance ration is always 1. The ratio = 1 / (0.5 ** steps_counter)
        if terminal:
            return pow(2, steps_counter)
        steps_counter += 1


def single_run(steps_number):
    """Single run for a given number os steps"""
    values = list()
    numerator = 0
    for i in trange(1, steps_number):

        numerator += generate_episode()

        values.append(numerator / i)

    return np.array(values)


def plot_fig5_4(values):
    """Plot graph"""
    fig, ax = plt.subplots(figsize=(16, 8))
    for value in values:
        ax.plot(value)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Episodes (log scale)')
    ax.set_ylabel('Mean squared error\n(average over 100 runs)')
    ax.set_ylabel(r'MC estimate of $v_{\pi}(s)$ with ordinary' '\nimportance sampling (ten runs)')

    plt.rc('mathtext', fontset="cm")
    plt.ylim(-0.1, 3)
    plt.xscale("log")
    plt.savefig('figs/fig5_4.png', dpi=300)


if __name__ == '__main__':

    t0 = time.perf_counter()
    # Use multiprocessing for 10 independent runs
    results = list()

    # Not enough RAM for 100_000_000 steps
    steps_number = int(1e7)
    with mp.Pool(mp.cpu_count()) as pool:
        results = np.array(pool.map(single_run, [steps_number] * 10))
    t1 = time.perf_counter()
    print(f'Done in {t1 - t0} sec')
    # Plotting
    plot_fig5_4(results)
