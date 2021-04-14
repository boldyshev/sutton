#!/usr/bin/env python3

"""Figure 5.2, page 100"""

import time
import random
import itertools

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# Blackjack simulator is the same as in figure 5.1
from fig5_1 import BlackJack


def generate_episode(policy):
    """Single episode simulation

    :param policy: Player's policy. Policy is a dictonary with keys being the player's states
    and values being True or False for Hit and Stick respectively
    :type policy: dict
    :return: Sequence of state-actions and rewards
    :rtype: tuple
    """

    # Initialize episode
    black_jack = BlackJack()
    player_state = black_jack.initial_player_state
    action = random.choice((True, False))
    state_action_sequence = [(player_state, action)]

    # Player allways hits below 12
    while black_jack.player_sum < 12:
        player_state = black_jack.player_hits()
        state_action_sequence = [(player_state, action)]

    # Player's move
    while action:
        player_state = black_jack.player_hits()

        # Stop the game if the player goes bust
        if black_jack.player_sum > 21:
            return state_action_sequence, black_jack.rewards[black_jack.player_sum, black_jack.dealer_sum]

        # Add new state to sequence only if the player hadn't gone bust
        action = policy[player_state]
        state_action_sequence.append((player_state, action))

    # Dealer's move
    black_jack.play_as_dealer()

    return state_action_sequence, black_jack.rewards[black_jack.player_sum, black_jack.dealer_sum]


def plot_policy_values(policy, values):
    """Plot policy and values heatmap

    :param policy: Policy
    :type policy: dict
    :param values: Values
    :type values: dict
    :return: None
    :rtype: None
    """

    # Convert dicts to numpy arrays
    values_arr = np.zeros((2, 10, 10))
    policy_arr = np.zeros((2, 10, 10))
    for x, y, z in values.keys():
        value = values[x, y, z]
        action = policy[x, y, z]
        x, y, z = int(x), y - 12, z - 1
        values_arr[x, y, z] = value
        policy_arr[x, y, z] = action

    # The sequence of the arrays to be plotted
    arrays = (policy_arr[1, :, :],
              values_arr[1, :, :],
              policy_arr[0, :, :],
              values_arr[0, :, :],)

    # Labels and titles, mathtext for pi anf v
    ylabels = ["Usable ace\nPlayer sum", "Player sum", "No usable ace\nPlayer sum", "Player sum"]
    plt.rc('mathtext', fontset="cm")
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

    plt.show()


def monte_carlo_exploring_starts():
    """Implementation of the Monte Carlo ES (Exploring Starts), page 99. Calculating of the average return
    change to incremental with respect of exercise 5.4"""

    # Initialize
    usable_ace = (True, False)
    player_sums = range(12, 22)
    dealer_showing = range(1, 11)
    states = tuple(itertools.product(usable_ace, player_sums, dealer_showing))
    # True for Hit, False for Stick
    actions = (True, False)

    state_actions = tuple(itertools.product(states, actions))
    # Count state-action occurences for incremental average calculation
    state_action_counts = dict((state, 0) for state in state_actions)
    q = dict((state_action, 0) for state_action in state_actions)
    policy = dict(((x, y, z), False) if y in (20, 21) else ((x, y, z), True) for (x, y, z) in states)

    t0 = time.perf_counter()
    episodes_number = int(5e5)

    # Loop over episodes
    for i in tqdm(range(episodes_number)):
        # print('\r', f'Episode {i + 1}', end='')
        state_action_sequence, reward = generate_episode(policy)

        # Loop for each step of episode
        for (state, action) in state_action_sequence:

            # Update counter
            state_action_counts[state, action] += 1

            # Incremental average
            q[state, action] += (reward - q[state, action]) / state_action_counts[state, action]

            # If q for hit (True) is greater, then choose to hit (True), otherwise choose to stick (False)
            policy[state] = q[state, True] > q[state, False]

    # Get state-values from action-values
    values = dict()
    for player_state in states:
        action = policy[player_state]
        values[player_state] = q[player_state, action]

    t1 = time.perf_counter()
    print(f'\nDone in {t1 - t0} s')

    plot_policy_values(policy, values)


if __name__ == '__main__':
    monte_carlo_exploring_starts()
