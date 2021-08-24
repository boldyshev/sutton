#!/usr/bin/env python3

"""Exercise 5.12, page 111"""

import copy
import time
import random
import itertools
import pickle

import gym
import numpy as np
import skimage.draw
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm


class RacetrackDemo(gym.Env):

    max_velocity = 4
    min_velocity = 0
    cell_type = {'out': 0,
                 'track': 1,
                 'start': 2,
                 'finish': 3,
                 'car': 4}

    # acceleration -1, 0 or 1 for both X and Y axes, total 9 actions
    possible_actions = np.array(tuple(itertools.product([-1, 0, 1], repeat=2)))

    def __init__(self, racetrack_csv):
        self.racetrack = np.genfromtxt(racetrack_csv, delimiter=',', dtype=int)
        self.observation_space = gym.spaces.MultiDiscrete(
            [*self.racetrack.shape, self.max_velocity + 1, self.max_velocity + 1])
        self.start_y, self.start_x = np.where(self.racetrack == self.cell_type['start'])
        self.position = np.array([0, 0])
        self.velocity = np.array([0, 0])
        self.state = *self.position, *self.velocity
        self.reset()

        self.state_actions = dict()
        self.restrict_actions()

        self.outside = False
        self.crossed_finish = False

    def restrict_actions(self):
        """Exclude the actions that lead to zero or above 5 velocity"""
        args = [range(i) for i in self.observation_space.nvec]

        # Negative velocity along Y axis results in car moving upward on the rendered racetrack
        args[2] = range(-4, 1)
        for state in itertools.product(*args, repeat=1):
            pos1, pos2, vel1, vel2 = state
            velocity = np.array([vel1, vel2])
            actions = list()
            for action in self.possible_actions:
                new_velocity = velocity + action

                # Velocity is less than 5
                y_in_limits = -self.max_velocity <= new_velocity[0] <= self.min_velocity
                x_in_limits = self.min_velocity <= new_velocity[1] <= self.max_velocity

                # Both velocity components can't be zero at a time
                both_non_zero = new_velocity.any()
                if y_in_limits and x_in_limits and both_non_zero:
                    actions.append(action)

            actions = np.array(actions)
            self.state_actions[state] = actions

    def reset(self):
        """Move car to the finish line"""
        x = np.random.choice(self.start_x)
        y = self.start_y[0]
        self.position = np.array([y, x])
        self.velocity = np.array([0, 0])
        self.state = *self.position, *self.velocity

    def reset_demo(self):
        """For rendering"""
        self.reset()
        self.outside = False
        self.crossed_finish = False

    def step(self, action):

        self.velocity += action

        previous_position = copy.deepcopy(self.position)
        self.position += self.velocity

        self.state = *self.position, *self.velocity

        # Get coordinates of projected path
        xs, ys = skimage.draw.line(*previous_position, *self.position)
        path = list(self.racetrack[xs, ys])

        # Check for intersections with finish line or track border
        crossed_finish = self.cell_type['finish'] in path
        if self.crossed_finish:
            idx = path.index(self.cell_type['finish'])
            path = path[:idx]
        outside = self.cell_type['out'] in path
        if outside or crossed_finish:
            self.reset()

        return self.state, crossed_finish

    def step_demo(self, action):
        """For rendering"""
        if self.outside or self.crossed_finish:
            self.reset_demo()
            return self.state, self.crossed_finish

        self.velocity += action

        previous_position = copy.deepcopy(self.position)
        self.position += self.velocity

        self.state = *self.position, *self.velocity

        xs, ys = skimage.draw.line(*previous_position, *self.position)
        path = list(self.racetrack[xs, ys])

        self.crossed_finish = self.cell_type['finish'] in path
        if self.crossed_finish:
            idx = path.index(self.cell_type['finish'])
            path = path[:idx]
        self.outside = self.cell_type['out'] in path

        return self.state, self.crossed_finish

    def render(self, save_pic=False, track=None, step=None):

        pic = copy.deepcopy(self.racetrack)
        pic[tuple(self.position)] = self.cell_type['car']
        color_map = mcolors.ListedColormap(['gray', 'white', 'red', 'green', 'blue'])

        plt.pcolor(pic, cmap=color_map, edgecolors='black', linewidths=0.5)

        ax = plt.gca()
        ax.set_aspect('equal')
        if not ax.yaxis_inverted():
            ax.invert_yaxis()
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)

        if save_pic:
            fig_name = f'./figs/fig{track}-{step}.svg'
            plt.savefig(fig_name, format='svg')
            return None

        plt.show()


class RacetrackNoisy(RacetrackDemo):

    def step(self, action):
        if random.random() < 0.1:
            action = np.zeros(2, dtype=int)
        result = super().step(action)

        return result


def generate_episode(env, policy):
    state = env.state
    state_action_idx_sequence = list()

    crossed_finish = False
    t0 = time.perf_counter()
    while not crossed_finish:

        action_idx = np.random.choice(range(len(env.state_actions[state])), p=policy[state])
        action = env.state_actions[state][action_idx]

        state_action_idx = state, action_idx
        state_action_idx_sequence.append(state_action_idx)

        state, crossed_finish = env.step(action)

    state_action_idx_sequence = state_action_idx_sequence

    t1 = time.perf_counter()


    return state_action_idx_sequence


def mc_on_policy_control(env, q, policy, episodes_number, eps=0.1):
    """On-policy first-visit MC control (for eps-soft policies), for estimating policies,
    section 5.4, p. 101"""

    state_action_counter = dict()
    action_numbers = dict()
    for state, actions in env.state_actions.items():
        actions_number = len(actions)
        action_numbers[state] = actions_number
        state_action_counter[state] = np.zeros(actions_number, dtype=int)

    for _ in tqdm(range(episodes_number)):
        _return = 0
        state_action_idx_sequence = generate_episode(env, policy)

        # Reverse the sequence to calculate the return for certain state-action pair
        state_action_idx_sequence = state_action_idx_sequence[::-1]

        # first-visit method
        state_action_idx_set = set(state_action_idx_sequence)

        for state_action_idx in state_action_idx_set:
            # As reward is always -1, the return is the index of state-action pair in reversed sequence
            _return = -(state_action_idx_sequence.index(state_action_idx) + 1)
            state, action_idx = state_action_idx
            state_action_counter[state][action_idx] += 1
            q[state][action_idx] += (_return - q[state][action_idx]) / state_action_counter[state][action_idx]
            optimal_action_idx = np.argmax(q[state])
            policy[state][:] = eps / action_numbers[state]
            policy[state][optimal_action_idx] = 1 - eps + eps / action_numbers[state]
    return q, policy


def mc_off_policy_control(env, q, behavior_policy, target_policy, episodes_number):
    """Off-policy MC control, for estimating policies, section 5.7, p. 111"""
    actions_number = len(env.possible_actions)
    state_action_dim = *env.observation_space.nvec, actions_number
    cumulative_sum = np.zeros(state_action_dim, dtype=int)

    for _ in tqdm(range(episodes_number)):

        state_action_idx_sequence = generate_episode(env, policy)
        _return = 0
        weight = 1

        for (state, action_idx) in state_action_idx_sequence:
            _return -= 1
            cumulative_sum[state][action_idx] += weight
            q[state][action_idx] += (_return - q[state][action_idx]) * weight / cumulative_sum[state][action_idx]
            optimal_action = np.argmax(q[state])
            target_policy[state] = optimal_action
            if action_idx != optimal_action:
                break
            weight = weight / behavior_policy[state][action_idx]

    return q, target_policy


def episode_demo(env, track, policy):
    env.reset_demo()
    step = 0
    env.render(save_pic=True, track=track, step=step)
    while not env.crossed_finish:
        step += 1
        state = env.state
        action = np.argmax(policy[state])
        action = env.state_actions[state][action]
        env.step_demo(action)
        env.render(save_pic=True, track=track, step=step)


if __name__ == '__main__':
    racetrack = './racetrack/fig5-5a.csv'
    env = RacetrackNoisy(racetrack)

    q = dict()
    policy = dict()
    for state, actions in env.state_actions.items():
        actions_number = len(actions)
        q[state] = np.zeros(actions_number)
        policy[state] = np.ones(actions_number) / actions_number

    powers = range(1, 7)
    for power in powers:
        episodes_num = 10 ** power
        q, policy = mc_on_policy_control(env, q, policy, episodes_num)

    episode_demo(env, '5-5a', policy)




































