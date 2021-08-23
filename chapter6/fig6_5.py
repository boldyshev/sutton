#!/usr/bin/env python3
"""Figure 6.5, page 135"""

import random
import multiprocessing as mp
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from tqdm.contrib.concurrent import process_map


ACTIONS_A = 2
ACTIONS_B = 10

INITIAL_Q = {'terminal': np.zeros(2),
             'a': np.zeros(ACTIONS_A),
             'b': np.zeros(ACTIONS_B)}

TRANSITIONS = {'a': ['b', 'terminal'], 'b': ['terminal'] * 10}


def random_argmax(arr):
    """Standart numpy argmax returns the first maximal element, so if there are several maximal elements
    in array the result will be biased"""

    # returns an array of bools, True = max element
    arr_bool_max = arr == arr.max()

    # Indies of max elements
    indices_of_max = np.flatnonzero(arr_bool_max)

    # Random index
    random_maximal_element = np.random.choice(indices_of_max)

    return random_maximal_element


def epsilon_greedy_policy(action_values, eps=0.1):
    """Epsilon greedy policy with random tie breaking  in argmax. Returns index of the chosen action"""

    if random.random() > eps:
        action = random_argmax(action_values)
    else:
        action = random.randint(0, len(action_values) - 1)

    return action


def q_learning(a, episodes=300, alpha=0.1):
    """Q-learning algorithm for a number of episodes

    :param a: placeholder parameter to make possible the usage of map while in multiprocessing
    :type a: None
    :return: How many times LEFT action from A was chosen in each episode
    :rtype: numpy.array
    """
    q = deepcopy(INITIAL_Q)
    left_actions_from_a = np.zeros(episodes, dtype=int)

    for episode in range(episodes):
        state = 'a'

        while state != 'terminal':
            action = epsilon_greedy_policy(q[state])
            if state == 'a':
                reward = 0
                if action == 0:
                    left_actions_from_a[episode] += 1
            else:
                reward = np.random.normal(-0.1, 1)
            next_state = TRANSITIONS[state][action]
            q[state][action] += alpha * (reward + max(q[next_state]) - q[state][action])
            state = next_state

    return left_actions_from_a


def double_q_learning(a, episodes=300, alpha=0.1, eps=0.1):
    """Double Q-learning algorithm for a number of episodes, page 136

    :param a: placeholder parameter to make possible the usage of map while in multiprocessing
    :type a: None
    :return: How many times LEFT action from A was chosen in each episode
    :rtype: numpy.array
    """

    q1 = deepcopy(INITIAL_Q)
    q2 = deepcopy(INITIAL_Q)

    left_actions_from_a = np.zeros(episodes, dtype=int)

    for episode in range(episodes):
        state = 'a'

        while state != 'terminal':
            q12_state = q1[state] + q2[state]
            action = epsilon_greedy_policy(q12_state)

            if state == 'a':
                reward = 0
                if action == 0:
                    left_actions_from_a[episode] += 1
            else:
                reward = np.random.normal(-0.1, 1)

            next_state = TRANSITIONS[state][action]

            if random.choice([True, False]):
                next_action = random_argmax(q1[next_state])
                q1[state][action] += alpha * (reward + q2[next_state][next_action] - q1[state][action])
            else:
                next_action = random_argmax(q2[next_state])
                q2[state][action] += alpha * (reward + q1[next_state][next_action] - q2[state][action])

            state = next_state

    return left_actions_from_a


def fig6_5():
    # This option is needed to get lower dispersion if run from Linux. Default for Windows 10 and MacOS.
    mp.set_start_method('spawn')

    runs = 10_000

    workers = mp.cpu_count()
    print('Q-learning')
    q = np.array(process_map(q_learning, range(runs), max_workers=workers, chunksize=1)).mean(axis=0)

    print('Double Q-learning')
    double_q = np.array(process_map(double_q_learning, range(runs), max_workers=workers, chunksize=1)).mean(axis=0)

    plt.plot(q * 100, label='Q-learning')
    plt.plot(double_q * 100, label='Double Q-learning')
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('% left actions from A')
    plt.show()


if __name__ == '__main__':
    fig6_5()
