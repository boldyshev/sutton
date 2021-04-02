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

# diamonds, clubs, hearts and spades
SUITS_NUMBER = 4

# jacks, queens and kings
FACE_CARDS_NUMBER = 3

# All scores above 21 counted as 0
FINAL_SCORES = tuple(itertools.product((0, 20, 21), list(range(17, 22)) + [0]))

# Precalculate rewards for possible outcomes
REWARDS = dict()
for score_player, score_dealer in FINAL_SCORES:
    if score_player > score_dealer:
        reward = 1
    elif score_player == score_dealer:
        reward = 0
    else:
        reward = -1
    REWARDS[score_player, score_dealer] = reward

# Fill the deck of cards
DECK = list()
for card in range(1, 11):
    DECK += [card] * SUITS_NUMBER

# Fill the deck with face cards
DECK += [10] * SUITS_NUMBER * FACE_CARDS_NUMBER


class BlackJack:

    def __init__(self, deck):

        # Create the deck
        self.deck = copy.copy(deck)

        # Begin game: deal two cards for dealer and player
        self.hand_player = [self.deal_card() for _ in range(2)]
        self.hand_dealer = [self.deal_card() for _ in range(2)]

        # Check if player has usable ace
        _usable_ace = self.usable_ace(self.hand_player)

        # Put one of the dealer's cards face up
        self.visible_dealer_card = self.hand_dealer[0]

        # Compute the scores
        self.score_dealer, self.hand_dealer = self.score(self.hand_dealer)
        self.score_player, self.hand_player = self.score(self.hand_player)

        # Initial state

        # Player always hits if his score is below 12
        while self.score_player < 12:
            card = self.deal_card()
            self.hand_player.append(card)
            self.score_player, self.hand_player = self.score(self.hand_player)
        self.player_states = [(self.score_player, self.visible_dealer_card, _usable_ace)]

    def deal_card(self):
        """Take a random card out of the deck"""
        return self.deck.pop(random.randrange(len(self.deck)))

    @staticmethod
    def usable_ace(hand):
        ace = 1 in hand
        usable = sum(hand) < 12
        return ace and usable

    def score(self, hand):
        """Compute the score of a given hand"""
        new_hand = copy.copy(hand)
        _score = sum(hand)
        _usable_ace = self.usable_ace(hand)
        _used_ace = 11 in hand
        if _usable_ace:
            ace_index = new_hand.index(1)
            new_hand[ace_index] = 11
            return _score + 10, new_hand
        elif _score > 21 and _used_ace:
            ace_index = new_hand.index(11)
            new_hand[ace_index] = 1
            return _score - 10, new_hand
        elif _score > 21:
            return 0, new_hand
        return _score, new_hand

    def play_as_dealer(self):
        while 2 < self.score_dealer < 17:
            card = self.deal_card()
            self.hand_dealer.append(card)
            self.score_dealer, self.hand_dealer = self.score(self.hand_dealer)

    def step_as_player(self):
        card = self.deal_card()
        self.hand_player.append(card)
        _usable_ace = self.usable_ace(self.hand_player)
        self.score_player, self.hand_player = self.score(self.hand_player)
        new_state = self.score_player, self.visible_dealer_card, _usable_ace
        self.player_states.append(new_state)
        return new_state


def plot_values(values_usable_ace, values_no_usable_ace):
    fig, axes = plt.subplots(2, 1, figsize=(6, 9))
    axes = axes.flatten()

    dealer_states = range(1, 11)
    player_states = range(12, 22)

    h1 = sns.heatmap(values_usable_ace, ax=axes[0], xticklabels=dealer_states, yticklabels=player_states)
    h1.invert_yaxis()
    h1.set_ylabel("Usable ace\nplayer's sum")
    h1.set_xlabel("dealer's card")

    h2 = sns.heatmap(values_no_usable_ace, ax=axes[1], xticklabels=dealer_states, yticklabels=player_states)
    h2.invert_yaxis()
    h2.set_ylabel("No usable ace\nplayer's sum")
    h2.set_xlabel("dealer's card")
    plt.show()


def fig5_1():
    t0 = time.perf_counter()
    episodes_number = int(5e5)

    values_usable_ace = np.zeros((10, 10))
    values_no_usable_ace = np.zeros((10, 10))

    returns_usable_ace = collections.defaultdict(list)
    returns_no_usable_ace = collections.defaultdict(list)

    for i in range(episodes_number):
        print('\r', f'Episode {i + 1}', end='')
        black_jack = BlackJack(DECK)
        while 2 < black_jack.score_player < 20 and black_jack.score_dealer <= 21:
            black_jack.step_as_player()
        if black_jack.score_player < 22:
            black_jack.play_as_dealer()
            _reward = REWARDS[black_jack.score_player, black_jack.score_dealer]
        else:
            _reward = 0

        for state in set(black_jack.player_states):
            player_sum, visible_card, usable_ace = state
            if usable_ace:
                returns_usable_ace[player_sum, visible_card].append(_reward)
            else:
                returns_no_usable_ace[player_sum, visible_card].append(_reward)

    for player_sum, visible_card in itertools.product(range(12, 22), range(1, 11)):
        x, y = player_sum - 12, visible_card - 1
        values_usable_ace[x, y] = np.array(returns_usable_ace[player_sum, visible_card]).mean()
        values_no_usable_ace[x, y] = np.array(returns_no_usable_ace[player_sum, visible_card]).mean()

    t1 = time.perf_counter()
    print(f'\nDone in {t1 - t0} s')

    plot_values(values_usable_ace, values_no_usable_ace)


if __name__ == '__main__':
    fig5_1()
