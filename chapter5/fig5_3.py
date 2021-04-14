#!/usr/bin/env python3

"""Figure 5.3, page 106"""

import copy
from random import getrandbits
import itertools
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt

from fig5_1 import BlackJack


class BlackJack53(BlackJack):
    """Redefine __init__ function to always start from a single state"""

    def __init__(self):

        # New deck for each game
        self.deck = copy.copy(self._deck)

        # Begin game: deal two cards for dealer and player
        # "Hand" is a list of cards that player or dealer hold
        ace_deuce = getrandbits(1)
        if ace_deuce:
            self.player_hand = [1, 2]
            self.deck.remove(1)
            self.deck.remove(2)
        else:
            self.player_hand = [1, 1, 1]
            self.deck.remove(1)
            self.deck.remove(1)
            self.deck.remove(1)

        self.dealer_hand = [2, self.deal_card()]

        # Put one of the dealer's cards face up
        self.dealer_showing = self.dealer_hand[0]

        # Compute the sums
        usable_ace, self.player_sum, self.player_hand = self.count_sum(self.player_hand)
        _, self.dealer_sum, self.dealer_hand = self.count_sum(self.dealer_hand)

        # Initial player state
        self.initial_player_state = usable_ace, self.player_sum, self.dealer_showing


def generate_episode():
    """Single episode simulation

    :return: Sequence of state-actions and rewards
    :rtype: tuple
    """

    # Initialize episode
    black_jack = BlackJack53()
    player_state = black_jack.initial_player_state

    # The same as random.choice((True, False)) but faster
    action = getrandbits(1)
    state_action_sequence = [(player_state, action)]

    # Player's move
    while action:
        player_state = black_jack.player_hits()

        # Stop the game if the player goes bust
        if black_jack.player_sum > 21:
            return state_action_sequence, black_jack.rewards[black_jack.player_sum, black_jack.dealer_sum]

        # Add new state to sequence only if the player hadn't gone bust
        action = getrandbits(1)
        state_action_sequence.append((player_state, action))

    # Dealer's move
    black_jack.play_as_dealer()

    return state_action_sequence, black_jack.rewards[black_jack.player_sum, black_jack.dealer_sum]


def importance_sampling_ratio(state_action_sequence, target_policy):
    """Importance-sampling ration for a given state-action sequence"""
    target_product = 1
    for state, action in state_action_sequence:
        if target_policy[state] != action:
            # The numerator is 0 so the ratio is 0
            return 0

    # Behavior policy takes both actions with 0.5 probability, any sequence on length n gives 0.5^n
    behavior_product = pow(0.5, len(state_action_sequence))

    return target_product / behavior_product


def calculate_values(target_policy):

    episodes_number = int(1e4)

    ordinary_importance_sampling = list()
    weighted_importance_sampling = list()

    numerator = 0
    denominator = 0

    for i in range(1, episodes_number):
        state_action_sequence, _return = generate_episode()
        ratio = importance_sampling_ratio(state_action_sequence, target_policy)

        numerator += ratio * _return
        denominator += ratio

        ordinary = numerator / i
        ordinary_importance_sampling.append(ordinary)

        if denominator != 0:
            weighted = numerator / denominator
        else:
            weighted = 0
        weighted_importance_sampling.append(weighted)

    return ordinary_importance_sampling, weighted_importance_sampling


def off_policy_estimation():

    usable_ace = (True, False)
    player_sums = range(12, 22)
    dealer_showing = range(1, 11)

    states = tuple(itertools.product(usable_ace, player_sums, dealer_showing))
    target_policy = dict(((x, y, z), 0) if y in (20, 21) else ((x, y, z), 1) for (x, y, z) in states)

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(calculate_values, [target_policy] * 100)

    ordinary_arr = list()
    weighted_arr = list()

    for ordinary, weighted in results:
        ordinary_arr.append(ordinary)
        weighted_arr.append(weighted)

    ordinary_arr = np.array(ordinary_arr)
    weighted_arr = np.array(weighted_arr)

    true_value = -0.27726

    mse_ordinary = np.square(true_value - ordinary_arr).mean(axis=0)
    mse_weighted = np.square(true_value - weighted_arr).mean(axis=0)

    return mse_ordinary, mse_weighted


def plot_mse(mse_ordinary, mse_weighted):
    fig, ax = plt.subplots()
    ax.plot(mse_ordinary, label='Ordinary importance sampling', color='green')
    ax.plot(mse_weighted, label='Weighted importance sampling', color='red')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Episodes (log scale)')
    ax.set_ylabel('Mean squared error\n(average over 100 runs)')

    plt.xscale("log")
    plt.ylim(-0.01, 5)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    mse_ordinary, mse_weighted = off_policy_estimation()
    plot_mse(mse_ordinary, mse_weighted)
