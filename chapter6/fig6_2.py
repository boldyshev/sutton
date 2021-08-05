#!/usr/bin/env python3
import matplotlib.pyplot as plt

from example6_2 import *


def td_episode_increment(state_seq, reward_seq, values):
    value_increments = np.zeros(len(STATES))

    for i, state in enumerate(state_seq[:-1]):
        reward = reward_seq[i]
        new_state = state_seq[i + 1]
        value_increments[state] += reward + values[new_state] - values[state]
    state = state_seq[-1]
    reward = reward_seq[-1]
    terminal_value = 0
    value_increments[state] += reward + terminal_value - values[state]

    return value_increments


def mc_episode_increment(state_seq, reward_seq, values):
    value_increments = dict.fromkeys(STATES, 0)
    return_sequence = np.cumsum(reward_seq[::-1])[::-1]
    for state, _return in zip(state_seq, return_sequence):
        value_increments[state] += _return - values[state]

    return value_increments


def batch_update(random_walk, method, episodes=100, runs=100, alpha=0.001):

    errors_averaged_over_runs = np.zeros(episodes)
    for _ in tqdm(range(runs)):
        values = np.array([[0.5] * 5])

        value_increments = dict.fromkeys(STATES, 0)
        states_batch = list()
        rewards_batch = list()
        errors = list()
        for _ in range(episodes):
            state_seq, reward_seq = random_walk.generate_episode()
            states_batch.append(state_seq)
            rewards_batch.append(reward_seq)

            while True:
                if method == 'td':
                    value_increments = td_episode_increment(state_seq, reward_seq, values)

                elif method == 'mc':
                    value_increments = mc_episode_increment(state_seq, reward_seq, values)

                converged = (np.array(list(value_increments.values())) < 1e-2).all()
                if converged:
                    break
                for key in values:
                    values[key] += alpha * value_increments[key]
            values_arr = np.array(list(values.values())[1: -1])
            errors.append(np.sqrt(np.sum(np.power(values_arr - TRUE_VALUES, 2)) / 5.0))
        errors_averaged_over_runs += np.array(errors)
    errors_averaged_over_runs /= runs

    return errors_averaged_over_runs


if __name__ == '__main__':

    random_walk = RandomWalk(STATE_ACTIONS, INIT_VALUES, PROBABILITIES, REWARDS, TERMINALS)

    td_errors = batch_update(random_walk, 'td')
    mc_errors = batch_update(random_walk, 'mc')

    plt.plot(td_errors, label='td')
    plt.plot(mc_errors, label='mc')
    plt.xlabel('Walks/Episodes')
    plt.ylabel('RMS error, averaged over states')
    plt.title('Batch Training')
    plt.legend()
    plt.show()