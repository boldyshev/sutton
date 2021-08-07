#!/usr/bin/env python3
"""Figure 6.2, page 127"""

import time

from tqdm import tqdm

from example6_2 import *


def converge_batch(batch, values, update_method, alpha, epsilon=1e-3):
    """Repeat iterating over state-reward sequences of the batch until the value converges. Allows using either
    Monte-Carlo or temporal-difference methods.

    :param batch: pairs of state and reward sequences
    :type batch: list
    :param values: state values
    :type values: numpy.array
    :param update_method: RandomWalk.td_batch_episode_increment or RandomWalk.mc_batch_episode_increment
    :type update_method: function
    :param alpha: constant step-size parameter
    :type alpha: float
    :param epsilon: estimating accuracy
    :type epsilon: float
    :return: values updated with a given method
    :rtype: numpy.array
    """

    while True:
        value_increments = np.zeros(len(STATES))

        for state_seq, reward_seq in batch:
            update_method(state_seq, reward_seq, values, value_increments)

        value_increments *= alpha
        converged = (np.absolute(value_increments) < epsilon).all()
        if converged:
            break
        values += value_increments

    return values


def batch_single_run(random_walk, update_method, episodes=100, alpha=0.001):
    """Single run over all episodes with batch including all previous episodes generated so far. Allows using either
    Monte-Carlo or temporal-difference methods.

    :param random_walk: Markov reward process with given parameters
    :type random_walk: RandomWalk object
    :param update_method: RandomWalk.td_batch_episode_increment or RandomWalk.mc_batch_episode_increment
    :type update_method: function
    :param episodes: numper of episodes
    :type episodes: int
    :param alpha: constant step-size parameter
    :type alpha: float
    :return: RMS value errors averaged over all states
    :rtype: numpy.array
    """
    values = copy.copy(INIT_VALUES)
    batch = list()
    errors = list()
    for _ in range(episodes):
        state_reward_seq = random_walk.generate_episode()
        batch.append(state_reward_seq)
        values = converge_batch(batch, values, update_method, alpha)
        errors.append(np.sqrt(((values[1: -1] - TRUE_VALUES) ** 2).mean()))

    return errors


def batch_update_single_thread(random_walk, update_method, episodes=100, runs=100, alpha=0.001):
    """Total batch update averaged over multiple runs. Single thread version. Allows using either Monte-Carlo or
    temporal-difference methods.

    :param random_walk: Markov reward process with given parameters
    :type random_walk: RandomWalk object
    :param update_method: RandomWalk.td_batch_episode_increment or RandomWalk.mc_batch_episode_increment
    :type update_method: function
    :param episodes: numper of episodes
    :type episodes: int
    :param runs: number of runs
    :type runs: int
    :param alpha: constant step-size parameter
    :type alpha: float
    :return: RMS value errors averaged over all runs
    :rtype: numpy.array
    """

    errors_arr = list()
    for _ in tqdm(range(runs)):
        errors_arr.append(batch_single_run(random_walk, update_method, episodes, alpha))

    errors_arr = np.mean(errors_arr, axis=0)

    return errors_arr


def batch_update_multi_thread(random_walk, update_method, episodes=100, runs=100, alpha=0.001):
    """Multi-thread version of batch update"""

    with mp.Pool(mp.cpu_count()) as pool:
        args = [(random_walk, update_method, episodes, alpha)] * runs
        errors = np.array(pool.starmap(batch_single_run, args))
    errors = np.mean(errors, axis=0)
    return errors


if __name__ == '__main__':

    random_walk = RandomWalk(STATE_ACTIONS, INIT_VALUES, PROBABILITIES, REWARDS, TERMINALS)

    mp.set_start_method('spawn')

    td_update_method = random_walk.td_batch_episode_increment
    mc_update_method = random_walk.mc_batch_episode_increment

    t0 = time.perf_counter()
    print('TD batch update...', end=' ')
    td_errors = batch_update_multi_thread(random_walk, td_update_method)
    t1 = time.perf_counter()
    print(f'Done in {t1 - t0} sec')

    print('MC batch update...', end=' ')
    mc_errors = batch_update_multi_thread(random_walk, mc_update_method)
    t2 = time.perf_counter()
    print(f'Done in {t2 - t1} sec')

    plt.plot(td_errors, label='td')
    plt.plot(mc_errors, label='mc')
    plt.xlabel('Walks/Episodes')
    plt.ylabel('RMS error, averaged over states')
    plt.title('Batch Training')
    plt.legend()
    plt.show()
