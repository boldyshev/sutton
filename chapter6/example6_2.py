#!/usr/bin/env python3

import copy
import time

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Create graph: vertices are states, edges are actions (transitions)
STATE_ACTIONS = {'left': [None, None],
                 'a': [('a', 'left'), ('a', 'b')],
                 'b': [('b', 'a'), ('b', 'c')],
                 'c': [('c', 'b'), ('c', 'd')],
                 'd': [('d', 'c'), ('d', 'e')],
                 'e': [('e', 'd'), ('e', 'right')],
                 'right': [None, None]}

# List of states
STATES = list(STATE_ACTIONS.keys())
TERMINALS = 'left', 'right'

# Transition probabilities
PROBABILITIES = dict.fromkeys(STATES, [0.5, 0.5])
PROBABILITIES.update({'left': [1], 'right': [1]})

# State values (probability to reach 'Right' state)
INIT_VALUES = dict.fromkeys(STATES, 0.5)
INIT_VALUES.update({'left': 0, 'right': 0})
TRUE_VALUES = np.arange(1, 6) / 6

# All actions
ACTIONS = list()
for key in 'abcde':
    ACTIONS += STATE_ACTIONS[key]

# Reward for each action
REWARDS = dict.fromkeys(ACTIONS, 0)
REWARDS[('e', 'right')] = 1


class RandomWalk:

    def __init__(self, graph, values, probabilities, rewards, terminals):

        self.graph = graph
        self.values = copy.deepcopy(values)

        self.probabilities = probabilities
        self.rewards = rewards
        self.terminals = terminals

    def step(self, state):

        actions = self.graph[state]
        probs = self.probabilities[state]
        action = actions[np.random.choice(len(actions), p=probs)]

        state = action[1]
        reward = self.rewards[action]

        return state, reward

    def generate_episode(self, state='c'):

        state_sequence = list()
        reward_sequence = list()

        while state not in self.terminals:
            state_sequence.append(state)
            state, reward = self.step(state)
            reward_sequence.append(reward)

        return state_sequence, reward_sequence

    def mc_estimate_episode(self, state='c', alpha=0.1):

        state_sequence, reward_sequence = self.generate_episode(state)
        return_sequence = np.cumsum(reward_sequence[::-1])[::-1]

        for state, _return in zip(state_sequence, return_sequence):
            self.values[state] += alpha * (_return - self.values[state])

        return self.values

    def td_estimate_episode(self, state='c', alpha=0.1):

        state_sequence, reward_sequence = self.generate_episode(state)

        for i, state in enumerate(state_sequence[:-1]):
            reward = reward_sequence[i]
            new_state = state_sequence[i + 1]
            self.values[state] += alpha * (reward + self.values[new_state] - self.values[state])
        state = state_sequence[-1]
        reward = reward_sequence[-1]
        terminal_value = 0
        self.values[state] += alpha * (reward + terminal_value - self.values[state])

        return self.values


def plot_estimated_value(random_walk, axs):

    # TD-estimated values, left figure
    for episodes_num in (0, 1, 10, 100):
        for _ in range(episodes_num):
            random_walk.td_estimate_episode()
        td_values = list(random_walk.values.values())[1: -1]
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

                vals = list(vals.values())[1: -1]
                values.append(vals)

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

    # RMS errors, right
    mc_alphas = 0.01, 0.02, 0.03, 0.04
    plot_rms_error(random_walk, axs[1], mc_alphas, 'mc')

    td_alphas = 0.05, 0.1, 0.15
    plot_rms_error(random_walk, axs[1], td_alphas, 'td')

    plt.show()
