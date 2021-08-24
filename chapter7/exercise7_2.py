#!/usr/bin/env python3
"""Exercise 7.2, page 143"""

import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt

import gym
import gym.utils.seeding

from tqdm.contrib.concurrent import process_map
from scipy import signal

# ns are powers of 2
# N_LIST = [2 ** power for power in range(10)]
N_LIST = [4]

# Set maximal alpha manually to get smoother curves
ALPHA_MAX_LIST = [1, 1, 1, 1, 1, 0.5, 0.3, 0.2, 0.2, 0.2]

ALPHA_MAX_LIST = [0.6]

# Alpha sampling rate
ALPHA_SAMPLING_RATE = 40


class RandomWalkEnv(gym.Env):

    def __init__(self, size=5):

        # All possible states including terminal
        self.observation_space = gym.spaces.Discrete(size + 2)

        # Terminal states are the left- and rightmost ones
        self.terminal_states = 0, self.observation_space.n - 1

        # Predetermine rewards to save time calculating step()
        self.rewards = np.zeros(self.observation_space.n, dtype=int)
        self.rewards[-1] = 1
        if size == 19:
            self.rewards[0] = -1

        # True values
        self.true_values = self.get_true_values()

        # Start form the middle state
        self.state = self.observation_space.n // 2

    def step(self):

        transition = np.random.choice([-1, 1])

        self.state += transition
        reward = self.rewards[self.state]

        return self.state, reward

    def reset(self):
        self.state = self.observation_space.n // 2
        return self.state

    def get_true_values(self, delta=1e-5):
        """Use dynamic programming to estimate true values. Iterative policy evaluation algorithm, page 75"""

        values = np.zeros(self.observation_space.n)
        updated_values = np.zeros(self.observation_space.n)
        non_terminal_states = range(1, self.observation_space.n - 1)

        converged = False
        while not converged:

            for state in non_terminal_states:
                values[state] = updated_values[state]

                left_state_value = values[state - 1] + self.rewards[state - 1]
                right_state_value = values[state + 1] + self.rewards[state + 1]
                updated_values[state] = 0.5 * (left_state_value + right_state_value)

            converged = sum(abs(values - updated_values)) < delta

        return values


gym.envs.registration.register(
    id='RandomWalk-v1',
    entry_point=lambda size: RandomWalkEnv(size),
    nondeterministic=True,
    kwargs={'size': 5}
)


def td_nstep_episode_estimate(values, random_walk, n, alpha):
    """n-step TD algorithm implementation, page 144. """

    t = 0
    terminal_step = np.inf

    # Use dicts for state and reward sequences to keep indexes like in the book
    state_seq, reward_seq = dict(), dict()

    # No reward precedes S(0)
    state_seq[t] = random_walk.state

    # Episode starts from the central state
    random_walk.reset()

    tau = -n + 1
    while tau != terminal_step - 1:

        if t < terminal_step:
            state_seq[t + 1], reward_seq[t + 1] = random_walk.step()
            if state_seq[t + 1] in random_walk.terminal_states:
                terminal_step = t + 1
        tau = t - n + 1
        if tau >= 0:
            _return = sum(reward_seq[i] for i in range(tau + 1, min(tau + n, terminal_step) + 1))
            if tau + n < terminal_step:
                _return += values[state_seq[tau + n]]
            values[state_seq[tau]] += alpha * (_return - values[state_seq[tau]])

        t += 1

    return values


def get_rms_error_for_alpha(random_walk, n, alpha, episodes, method):
    """Get RMS error for single value of n and alpha"""

    values = np.zeros(random_walk.observation_space.n)
    errors_by_episodes = list()
    for _ in range(episodes):
        # Estimate values with TD n-step method
        values = method(values, random_walk, n, alpha)

        # Get averaged squared error between the predictions and their true values
        rms_error = np.sqrt(np.power(values[1: -1] - random_walk.true_values[1: -1], 2).mean())

        # Store error for this episode
        errors_by_episodes.append(rms_error)

    # Average over all episodes
    error_alpha = np.mean(errors_by_episodes)

    return error_alpha


def get_errors_for_n(random_walk, n, alpha_max, method, episodes=10):
    """Get errors for all alphas single n"""

    alphas = np.linspace(0, alpha_max, ALPHA_SAMPLING_RATE)
    errors_by_alphas = list()
    for alpha in alphas:
        error = get_rms_error_for_alpha(random_walk, n, alpha, episodes, method)
        errors_by_alphas.append(error)

    return np.stack(errors_by_alphas)


def errors_single_run(method):
    """10 arrays of errors for each n. Averaged over 19 states and 10 episodes"""

    random_walk = gym.make('RandomWalk-v1', size=19)
    errors_by_n = list()
    for n, alpha_max in zip(N_LIST, ALPHA_MAX_LIST):
        errors = get_errors_for_n(random_walk, n, alpha_max, method)
        errors_by_n.append(errors)
    return np.stack(errors_by_n)


def average_over_runs(method, runs=100):
    """Use multiprocessing to average errors over 100 runs"""

    workers = mp.cpu_count()
    errors_runs = np.array(process_map(errors_single_run, [method] * runs, max_workers=workers, chunksize=1))
    averaged_errors = errors_runs.mean(axis=0)

    return averaged_errors


def td_error_episode_estimate(values, random_walk, n, alpha):
    """n-step TD algorithm implementation, page 144. """

    t = 0
    terminal_step = np.inf
    td_error = np.zeros_like(values)

    # Use dicts for state and reward sequences to keep indexes like in the book
    state_seq, reward_seq = dict(), dict()

    # No reward precedes S(0)
    state_seq[t] = random_walk.state

    # Episode starts from the central state
    random_walk.reset()

    tau = -n + 1
    while tau != terminal_step - 1:

        if t < terminal_step:
            state_seq[t + 1], reward_seq[t + 1] = random_walk.step()
            if state_seq[t + 1] in random_walk.terminal_states:
                terminal_step = t + 1
        tau = t - n + 1
        if tau >= 0:
            _return = sum(reward_seq[i] for i in range(tau + 1, min(tau + n, terminal_step) + 1))
            if tau + n < terminal_step:
                _return += values[state_seq[tau + n]]
            td_error += alpha * (_return - values[state_seq[tau]])

        t += 1
    values += td_error

    return values


def exercise7_2():
    # This option is needed to get lower dispersion if run from Linux. Default for Windows 10 and MacOS.
    mp.set_start_method('spawn')

    runs = 100

    errors_by_n1 = average_over_runs(td_error_episode_estimate, runs=runs)
    errors_by_n2 = average_over_runs(td_nstep_episode_estimate, runs=runs)

    for i, (errors1, errors2) in enumerate(zip(errors_by_n1, errors_by_n2)):
        # Use Savitzky-Golay filter to smooth errors
        errors_smooth1 = signal.savgol_filter(errors1, 19, 4)
        errors_smooth2 = signal.savgol_filter(errors2, 19, 4)

        alphas = np.linspace(0, ALPHA_MAX_LIST[i], ALPHA_SAMPLING_RATE)
        plt.plot(alphas, errors_smooth1, label='TD error accumulate')
        plt.plot(alphas, errors_smooth2, label='Each step update')

    # plt.xticks(np.linspace(0, 1, 6))
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Average RMS error over 19 states and first 10 episodes')
    plt.legend()
    # plt.ylim(0.25, 0.55)
    plt.show()
    # plt.savefig('fig7_2.png')


if __name__ == '__main__':
    exercise7_2()
