#!/usr/bin/env python3
"""Figure 7.4, page 147"""

import random
from collections import defaultdict
import multiprocessing as mp

import numpy as np
import gym
import skimage.draw
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


class Gridworld(gym.Env):

    # 0 - UP, 1 - DOWN, 2 - LEFT, 3 - RIGHT
    actions = {0: np.array([0, 1]),
               1: np.array([0, -1]),
               2: np.array([-1, 0]),
               3: np.array([1, 0])}

    def __init__(self, size=(10, 8)):

        # Possible states
        self.observation_space = gym.spaces.MultiDiscrete(size)

        # Possible actions: up, down, left, right
        self.action_space = gym.spaces.Discrete(4)

        # Start coordinates x,y
        self.start = 1, 3

        # Goal coordinates x, y
        self.goal = 6, 3

        # All rewards are zero except when action leads to the goal state
        self.rewards = defaultdict(int)
        self.rewards[self.goal] = 1

        # Current state
        self.state = np.array([1, 3])

    def step(self, action):

        move = self.actions[action]
        self.state += move

        # Keep x, y position inside the world
        self.state[0] = np.clip(self.state[0], 0, self.observation_space.nvec[0] - 1)
        self.state[1] = np.clip(self.state[1], 0, self.observation_space.nvec[1] - 1)

        state = tuple(self.state)
        reward = self.rewards[tuple(self.state)]

        return state, reward

    def reset(self):
        self.state = np.array(self.start)

    def render(self):
        world_dimensions = self.observation_space.nvec
        gridworld = np.zeros(world_dimensions, dtype=int)
        color_map = mcolors.ListedColormap(['white'])
        plt.pcolor(gridworld.T, cmap=color_map, edgecolors='black', linewidths=0.5)
        ax = plt.gca()
        ax.text(self.goal[0] + 0.5, self.goal[1] + 0.5, 'G', ha="center", va="center", fontsize=24)
        ax.set_aspect('equal')
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)

    def render_path(self, path):
        plt.figure()

        self.render()
        for state1, state2 in zip(path[:-1], path[1:]):
            x = np.array([state1[0], state2[0]]) + 0.5
            y = np.array([state1[1], state2[1]]) + 0.5
            plt.plot(x, y, color=u'#1f77b4', linewidth=2)
        plt.title("Path taken", color=u'#1f77b4', fontsize=24)
        plt.savefig('figs/fig7_4_path.svg', format='svg')

    def render_updates(self, q_updates, n):
        plt.figure()

        self.render()
        for (x, y, a) in q_updates:
            action = self.actions[a]
            dx = np.clip(action[0], -0.4, 0.4)
            dy = np.clip(action[1], -0.4, 0.4)
            plt.arrow(x + 0.5, y + 0.5, dx, dy, color=u'#1f77b4', linewidth=2, head_width=0.1)
        plt.title(f"Action values increased \nby {n}-step Sarsa", color=u'#1f77b4', fontsize=18)

        plt.savefig(f'figs/fig7_4_updates_{n}.svg', format='svg')


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


def sarsa_nstep_episode_estimate(env, q, n, actions, alpha=0.5):
    """n-step Sarsa algorithm implementation (page 147) adatped for a given sequence of actions"""

    env.reset()
    t = 0
    terminal_step = np.inf
    # Use dicts for state and reward sequences to keep indexes like in the book
    state_seq, reward_seq, action_seq = dict(), dict(), dict()

    # Initialize and store S_0 != terminal
    x, y = tuple(env.state)
    state_seq[t] = (x, y)

    # Select and store an action A_0
    action = actions[t]
    action_seq[t] = action
    q_updates = list()

    tau = -n + 1
    while tau != terminal_step - 1:

        if t < terminal_step:
            # Take action A_t, observe and store S_{t+1} and R_{t+1}
            state_seq[t + 1], reward_seq[t + 1] = env.step(action_seq[t])

            if state_seq[t + 1] == env.goal:
                terminal_step = t + 1
            else:
                # Select and store an action A_{t+1}
                action_seq[t + 1] = actions[t + 1]

        tau = t - n + 1
        if tau >= 0:
            _return = sum(reward_seq[i] for i in range(tau + 1, min(tau + n, terminal_step) + 1))
            if tau + n < terminal_step:
                (x, y), a = state_seq[tau + n], action_seq[tau + n]
                _return += q[x, y, a]

            (x, y), a = state_seq[tau], action_seq[tau]
            q_increment = alpha * (_return - q[x, y, a])
            q[x, y, a] += q_increment
            if q_increment > 0:
                q_updates.append((x, y, a))

        t += 1

    # Get path as list of sequent states
    path = list(state_seq.values())

    return q, path, q_updates


def sarsa_nstep(env, n, episodes):
    q_dimensions = *env.observation_space.nvec, env.action_space.n
    q = np.zeros(q_dimensions)
    for _ in tqdm(range(episodes)):
        q, path, q_updates = sarsa_nstep_episode_estimate(env, q, n)

    return path, q_updates


def fig7_4():
    env = Gridworld()
    actions = [3, 3, 0, 3, 0, 0, 3, 3, 3, 1, 1, 3, 1, 1, 2, 2, 0]

    n = 1
    q_dimensions = *env.observation_space.nvec, env.action_space.n
    q = np.zeros(q_dimensions)
    q, path, q_updates = sarsa_nstep_episode_estimate(env, q, n, actions)
    env.render_path(path)
    env.render_updates(q_updates, n)

    n = 10
    env.reset()
    q = np.zeros(q_dimensions)
    q, path, q_updates = sarsa_nstep_episode_estimate(env, q, n, actions)
    env.render_updates(q_updates, n)


if __name__ == '__main__':
    fig7_4()
