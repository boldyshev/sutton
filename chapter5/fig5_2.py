#!/usr/bin/env python3

"""Figure 5.2, page 100"""

import time
import random
import itertools

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from fig5_1 import BlackJack


def generate_episode(policy):
    black_jack = BlackJack()
    player_state = black_jack.initial_player_state
    action = random.choice((True, False))
    state_action_sequence = [(player_state, action)]

    while black_jack.player_sum < 12:
        player_state = black_jack.player_hits()
        state_action_sequence = [(player_state, action)]

    while action:
        player_state = black_jack.player_hits()

        if black_jack.player_sum > 21:
            return state_action_sequence, black_jack.rewards[black_jack.player_sum, black_jack.dealer_sum]

        action = policy[player_state]
        state_action_sequence.append((player_state, action))

    black_jack.play_as_dealer()

    return state_action_sequence, black_jack.rewards[black_jack.player_sum, black_jack.dealer_sum]


def plot_policy_values(policy, values):
    values_arr = np.zeros((2, 10, 10))
    policy_arr = np.zeros((2, 10, 10))
    for x, y, z in values.keys():
        value = values[x, y, z]
        action = policy[x, y, z]
        x, y, z = int(x), y - 12, z - 1
        values_arr[x, y, z] = value
        policy_arr[x, y, z] = action

    arrays = (policy_arr[1, :, :],
              values_arr[1, :, :],
              policy_arr[0, :, :],
              values_arr[0, :, :],)

    ylabels = ["Usable ace\nPlayer sum", "Player sum", "No usable ace\nPlayer sum", "Player sum"]
    titles = [r'$\pi_*$', r'$v_*$', '', '']

    _, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    dealer_showing = range(1, 11)
    player_sum = range(12, 22)

    subplots = list()
    for i, array in enumerate(arrays):
        subplot = sns.heatmap(array, ax=axes[i], xticklabels=dealer_showing, yticklabels=player_sum)
        subplots.append(subplot)
        subplot.invert_yaxis()
        subplot.set_ylabel(ylabels[i])
        subplot.set_xlabel("Dealer showing")
        subplot.set_title(titles[i])

    plt.rc('mathtext', fontset="cm")
    plt.show()


def monte_carlo_exploring_starts():
    usable_ace = (True, False)
    player_sums = range(12, 22)
    dealer_showing = range(1, 11)
    states = tuple(itertools.product(usable_ace, player_sums, dealer_showing))
    actions = (True, False)

    state_actions = tuple(itertools.product(states, actions))
    state_action_counts = dict((state, 0) for state in state_actions)
    q = dict((state_action, 0) for state_action in state_actions)
    policy = dict(((x, y, z), False) if y in (20, 21) else ((x, y, z), True) for (x, y, z) in states)

    t0 = time.perf_counter()
    episodes_number = int(5e5)

    for i in range(episodes_number):
        print('\r', f'Episode {i + 1}', end='')
        state_action_sequence, reward = generate_episode(policy)
        for (state, action) in state_action_sequence:
            state_action_counts[state, action] += 1
            q[state, action] += (reward - q[state, action]) / state_action_counts[state, action]
            policy[state] = q[state, True] > q[state, False]

    values = dict()
    for player_state in states:
        action = policy[player_state]
        values[player_state] = q[player_state, action]

    t1 = time.perf_counter()
    print(f'\nDone in {t1 - t0} s')

    plot_policy_values(policy, values)


if __name__ == '__main__':
    monte_carlo_exploring_starts()
