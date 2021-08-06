#!/usr/bin/env python3

import copy
import time

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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
REWARDS = np.zeros((len(STATES), 2))
REWARDS[5, 1] = 1


class RandomWalk:

    def __init__(self, graph, values, probabilities, rewards, terminals):
        """Map all states to numeric arrays"""

        state_names = list(graph.keys())
        state_to_index = dict([(state, idx) for idx, state in enumerate(state_names)])
        self.states = [state_to_index[state] for state in state_names]
        self.terminals = [state_to_index[state] for state in state_names if state in terminals]

        self.actions = list()
        for actions in graph.values():
            action_idxs = [state_to_index[state] for state in actions]
            self.actions.append(action_idxs)

        self.values = copy.deepcopy(values)
        self.probabilities = probabilities
        self.rewards = rewards

    def step(self, state):
        next_state_idxs = range(len(self.actions[state]))
        next_state_idx = np.random.choice(next_state_idxs, p=self.probabilities[state])
        reward = self.rewards[state][next_state_idx]
        next_state = self.states[next_state_idx]

        return next_state, reward

    def generate_episode(self, state=3):

        state_sequence = list()
        reward_sequence = list()

        while state not in self.terminals:
            state_sequence.append(state)
            state, reward = self.step(state)
            reward_sequence.append(reward)

        return state_sequence, reward_sequence

    def mc_estimate_episode(self, state=3, alpha=0.1):

        state_sequence, reward_sequence = self.generate_episode(state)
        return_sequence = np.cumsum(reward_sequence[::-1])[::-1]

        for state, _return in zip(state_sequence, return_sequence):
            self.values[state] += alpha * (_return - self.values[state])

        return self.values

    def td_estimate_episode(self, state=3, alpha=0.1):

        while state not in self.terminals:
            next_state, reward = self.step(state)
            self.values[state] += alpha * (reward + self.values[next_state] - self.values[state])
            state = next_state

        return self.values


def plot_estimated_value(random_walk, axs):

    # TD-estimated values, left figure
    for episodes_num in (0, 1, 10, 100):
        for _ in range(episodes_num):
            random_walk.td_estimate_episode()
        td_values = random_walk.values[1: -1]
        axs.plot(STATES[1: -1], td_values, label=episodes_num)

    axs.plot(STATES[1: -1], TRUE_VALUES, label='True values')
    axs.legend()
    axs.set_xlabel('State')
    axs.set_ylabel('Estimated value')


def plot_rms_error(random_walk, axs, alphas, method, episodes=100, runs=100):
    for alpha in alphas:
        print(f'RMS for {method}, alpha = {alpha}')
        time.sleep(0.05)
        average_errors_over_runs = np.zeros(episodes + 1)
        for _ in tqdm(range(runs)):
            random_walk.values = copy.deepcopy(INIT_VALUES)
            values = [[0.5] * 5]
            for _ in range(episodes):
                if method == 'td':
                    vals = random_walk.td_estimate_episode(alpha=alpha)
                elif method == 'mc':
                    vals = random_walk.mc_estimate_episode(alpha=alpha)

                values.append(vals[1: -1])

            errors = np.sqrt(((values - TRUE_VALUES) ** 2).mean(axis=1))
            average_errors_over_runs += errors

        average_errors_over_runs /= runs
        axs.plot(average_errors_over_runs, label=f'{method} ' + r'$\alpha=$' + f'{alpha}')

    axs.legend()
    axs.set_xlabel('Walks/Episode')
    axs.set_ylabel('Empirical RMS error averaged over states')


if __name__ == '__main__':

    random_walk = RandomWalk(STATE_ACTIONS, INIT_VALUES, PROBABILITIES, REWARDS, TERMINALS)

    # Example 6.2 figure
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # TD-estimated values, left
    plot_estimated_value(random_walk, axs[0])

    # # RMS errors, right
    # mc_alphas = 0.01, 0.02, 0.03, 0.04
    # plot_rms_error(random_walk, axs[1], mc_alphas, 'mc')
    #
    # td_alphas = 0.05, 0.1, 0.15
    # plot_rms_error(random_walk, axs[1], td_alphas, 'td')

    plt.show()
