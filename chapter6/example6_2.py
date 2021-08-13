#!/usr/bin/env python3
"""Example 6.2, page 125"""


import copy
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt

# Create graph: vertices are states, edges are actions (transitions)
STATE_ACTIONS = {'left': ('left', 'left'),
                 'a': ('left', 'b'),
                 'b': ('a', 'c'),
                 'c': ('b', 'd'),
                 'd': ('c', 'e'),
                 'e': ('d', 'right'),
                 'right': ('right', 'right')}

# List of states
STATES = list(STATE_ACTIONS.keys())
TERMINALS = 'left', 'right'

# Transition probabilities
PROBABILITIES = np.full((len(STATES), 2), [0.5, 0.5])

# State values (probability to reach 'Right' state)
INIT_VALUES = np.full(len(STATES), 0.5)
np.put(INIT_VALUES, [0, -1], 0)
TRUE_VALUES = np.arange(1, 6) / 6


# Reward for each action
REWARDS = np.zeros((len(STATES), 2), dtype=int)
REWARDS[5, 1] = 1


class RandomWalk:
    """Represents Markov reward process defined by arbitrary graph"""

    def __init__(self, graph, values, probabilities, rewards, terminals):
        """Map states to numebers"""

        state_names = list(graph.keys())
        state_to_index = dict([(state, idx) for idx, state in enumerate(state_names)])

        # left, a, b, c, d, e, right -> 0, 1, 2, 3, 4, 5, 6
        self.states = [state_to_index[state] for state in state_names]
        self.terminals = [state_to_index[state] for state in state_names if state in terminals]

        # (left, b), ... -> [0, 2], ...
        self.actions = list()
        for actions in graph.values():
            action_idxs = [state_to_index[state] for state in actions]
            self.actions.append(action_idxs)

        self.values = copy.copy(values)
        self.probabilities = probabilities
        self.rewards = rewards

    def step(self, state):
        """Single step of the Markov reward process"""

        # Choose next state index
        next_state_idxs = range(len(self.actions[state]))
        next_state_idx = np.random.choice(next_state_idxs, p=self.probabilities[state])

        # Get next state and reward
        next_state = self.actions[state][next_state_idx]
        reward = self.rewards[state][next_state_idx]

        return next_state, reward

    def generate_episode(self, state=3):
        """Generates sequences of states and rewards, default starting state is C.
        Returns pairs (S_0, R_1), (S_1, R_2), ... . Terminal state is omitted"""
        state_sequence = list()
        reward_sequence = list()

        while state not in self.terminals:
            state_sequence.append(state)
            state, reward = self.step(state)
            reward_sequence.append(reward)

        return state_sequence, reward_sequence

    def mc_episode_estimate(self, state=3, alpha=0.1):
        """Estimate single episode" with Monte-Carlo method"""

        state_sequence, reward_sequence = self.generate_episode(state)
        return_sequence = np.cumsum(reward_sequence[::-1])[::-1]

        for state, _return in zip(state_sequence, return_sequence):
            self.values[state] += alpha * (_return - self.values[state])

        return self.values

    def td_episode_estimate(self, state=3, alpha=0.1):
        """Estimate single episode" with temporal-difference method"""

        while state not in self.terminals:
            next_state, reward = self.step(state)
            self.values[state] += alpha * (reward + self.values[next_state] - self.values[state])
            state = next_state

        return self.values

    @staticmethod
    def mc_batch_episode_increment(state_seq, reward_seq, values, value_increments):

        return_sequence = np.cumsum(reward_seq[::-1])[::-1]
        for state, _return in zip(state_seq, return_sequence):
            value_increments[state] += _return - values[state]

        return value_increments

    @staticmethod
    def td_batch_episode_increment(state_seq, reward_seq, values, value_increments):

        for i, state in enumerate(state_seq[:-1]):
            reward = reward_seq[i]
            new_state = state_seq[i + 1]
            value_increments[state] += reward + values[new_state] - values[state]
        state = state_seq[-1]
        reward = reward_seq[-1]
        terminal_value = 0
        value_increments[state] += reward + terminal_value - values[state]

        return value_increments


def plot_estimated_value(random_walk, axs):

    # TD-estimated values, left figure
    for episodes_num in (0, 1, 10, 100):
        for _ in range(episodes_num):
            random_walk.td_episode_estimate()
        td_values = random_walk.values[1: -1]
        axs.plot(STATES[1: -1], td_values, label=episodes_num)

    axs.plot(STATES[1: -1], TRUE_VALUES, label='True values')
    axs.legend()
    axs.set_xlabel('State')
    axs.set_ylabel('Estimated value')


def rms_single_run(random_walk, method, alpha, episodes=100):
    random_walk.values = copy.copy(INIT_VALUES)
    values = [[0.5] * 5]
    for _ in range(episodes):
        if method == 'td':
            vals = random_walk.td_episode_estimate(alpha=alpha)
        elif method == 'mc':
            vals = random_walk.mc_episode_estimate(alpha=alpha)

        vals = copy.copy(vals[1: -1])
        values.append(vals)

    errors = np.sqrt(((values - TRUE_VALUES) ** 2).mean(axis=1))

    return errors


def plot_rms_error(random_walk, axs, alphas, method, episodes=100, runs=100):
    for alpha in alphas:
        print('\r', f'RMS for {method}, alpha = {alpha}', end='')
        # Calculate runs in parallel
        with mp.Pool(mp.cpu_count()) as pool:
            args = [(random_walk, method, alpha, episodes)] * runs
            errors = np.array(pool.starmap(rms_single_run, args))

        errors = np.mean(errors, axis=0)
        axs.plot(errors, label=f'{method} ' + r'$\alpha=$' + f'{alpha}')
    print()
    axs.legend()
    axs.set_xlabel('Walks/Episode')
    axs.set_ylabel('Empirical RMS error averaged over states')


if __name__ == '__main__':

    random_walk = RandomWalk(STATE_ACTIONS, INIT_VALUES, PROBABILITIES, REWARDS, TERMINALS)

    # Example 6.2 figure
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # TD-estimated values, left
    plot_estimated_value(random_walk, axs[0])

    mp.set_start_method('spawn')

    # RMS errors, right
    mc_alphas = 0.01, 0.02, 0.03, 0.04
    plot_rms_error(random_walk, axs[1], mc_alphas, 'mc')

    td_alphas = 0.05, 0.1, 0.15
    plot_rms_error(random_walk, axs[1], td_alphas, 'td')

    plt.show()
