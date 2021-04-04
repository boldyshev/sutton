#!/usr/bin/env python3

"""Figure 5.1, page 94"""

import copy
import time
import random
import itertools
import collections

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class BlackJack:
    # diamonds, clubs, hearts and spades
    SUITS_NUMBER = 4

    # jacks, queens and kings
    FACE_CARD_TYPES = 3

    # Fill the deck with aces (1) and pip cards (2-10)
    _deck = [card for card in range(1, 11)] * SUITS_NUMBER

    # Fill the deck with face cards
    _deck += [10] * SUITS_NUMBER * FACE_CARD_TYPES

    def __init__(self):

        # New deck for each game
        self.deck = copy.copy(self._deck)

        # Begin game: deal two cards for dealer and player
        self.player_hand = [self.deal_card() for _ in range(2)]
        self.dealer_hand = [self.deal_card() for _ in range(2)]

        # Put one of the dealer's cards face up
        self.dealer_showing = self.dealer_hand[0]

        # Compute the sums
        usable_ace, self.player_sum, self.player_hand = self.count_sum(self.player_hand)
        _, self.dealer_sum, self.dealer_hand = self.count_sum(self.dealer_hand)

        self.initial_player_state = usable_ace, self.player_sum, self.dealer_showing

    def deal_card(self):
        """Take a random card from the deck"""
        return self.deck.pop(random.randrange(len(self.deck)))

    @staticmethod
    def usable_ace(hand):
        ace = 1 in hand
        usable = sum(hand) < 12
        return ace and usable

    def count_sum(self, hand):
        """Compute the sum of a given hand"""
        new_hand = copy.copy(hand)
        _sum = sum(hand)
        usable_ace = self.usable_ace(hand)
        used_ace = 11 in hand
        if usable_ace:
            ace_index = new_hand.index(1)
            new_hand[ace_index] = 11
            _sum += 10
        elif _sum > 21 and used_ace:
            ace_index = new_hand.index(11)
            new_hand[ace_index] = 1
            _sum -= 10
        return usable_ace, _sum, new_hand

    def play_as_dealer(self):
        while 2 < self.dealer_sum < 17:
            card = self.deal_card()
            self.dealer_hand.append(card)
            _, self.dealer_sum, self.dealer_hand = self.count_sum(self.dealer_hand)

    def player_hits(self):
        card = self.deal_card()
        self.player_hand.append(card)
        usable_ace, self.player_sum, self.player_hand = self.count_sum(self.player_hand)
        new_state = usable_ace, self.player_sum, self.dealer_showing
        return new_state


def generate_episode(policy):
    black_jack = BlackJack()
    player_state = black_jack.initial_player_state
    state_sequence = [player_state]

    if 21 in (black_jack.player_sum, black_jack.dealer_sum):
        if black_jack.player_sum == black_jack.dealer_sum:
            reward = 0
        elif black_jack.player_sum == 21:
            reward = 1
        elif black_jack.dealer_sum == 21:
            reward = -1
            if black_jack.player_sum < 12:
                state_sequence = list()
        return state_sequence, reward

    while black_jack.player_sum < 12:
        player_state = black_jack.player_hits()
        state_sequence = [player_state]
        if black_jack.player_sum == 21:
            reward = 1
            return state_sequence, reward

    while policy[player_state]:
        player_state = black_jack.player_hits()
        if black_jack.player_sum > 21:
            reward = -1
            return state_sequence, reward
        elif black_jack.player_sum == 21:
            reward = 1
            state_sequence.append(player_state)
            return state_sequence, reward

        state_sequence.append(player_state)

    black_jack.play_as_dealer()
    if black_jack.dealer_sum == 21:
        reward = -1
        return state_sequence, reward
    elif black_jack.dealer_sum > 21:
        reward = 1
        return state_sequence, reward

    unequal_sum = black_jack.player_sum != black_jack.dealer_sum
    reward = int(unequal_sum)

    return state_sequence, reward


def plot_values(values):
    values_arr = np.zeros((2, 10, 10))
    for (x, y, z), value in values.items():
        x, y, z = int(x), y - 12, z - 1
        values_arr[x, y, z] = value
    fig, axes = plt.subplots(2, 1, figsize=(6, 9))
    axes = axes.flatten()

    dealer_showing = range(1, 11)
    player_sum = range(12, 22)

    h1 = sns.heatmap(values_arr[1, :, :], ax=axes[0], xticklabels=dealer_showing, yticklabels=player_sum)
    h1.invert_yaxis()
    h1.set_ylabel("Usable ace\nPlayer sum")
    h1.set_xlabel("Dealer showing")

    h2 = sns.heatmap(values_arr[0, :, :], ax=axes[1], xticklabels=dealer_showing, yticklabels=player_sum)
    h2.invert_yaxis()
    h2.set_ylabel("No usable ace\nPlayer sum")
    h2.set_xlabel("Dealer showing")
    plt.show()


def first_visit_monte_carlo_prediction():
    usable_ace = (True, False)
    player_sums = range(12, 22)
    dealer_showing = range(1, 11)
    states = tuple(itertools.product(usable_ace, player_sums, dealer_showing))

    values = dict((state, 0) for state in states)
    returns = dict((state, []) for state in states)
    policy = dict(((x, y, z), False) if y in (20, 21) else ((x, y, z), True) for (x, y, z) in states)

    t0 = time.perf_counter()
    episodes_number = int(1e4)

    for i in range(episodes_number):
        print('\r', f'Episode {i + 1}', end='')
        state_sequence, reward = generate_episode(policy)
        for state in state_sequence:
            returns[state].append(reward)

    for state in states:
        values[state] = np.array(returns[state]).mean()

    t1 = time.perf_counter()
    print(f'\nDone in {t1 - t0} s')

    plot_values(values)


if __name__ == '__main__':
    first_visit_monte_carlo_prediction()
