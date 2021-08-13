#!/usr/bin/env python3
"""Exercise 7.2, page 143"""

import copy


def td_nstep_episode_estimate(random_walk, state, n, alpha, terminal_step, values):

    tau = -n + 1
    t = 0

    state_seq, reward_seq = dict(), dict()
    while tau != terminal_step - 1:
        if t < terminal_step:
            next_state, next_reward = random_walk.step(state)
            state_seq[t + 1], reward_seq[t + 1] = next_state, next_reward
            if next_state in random_walk.terminals:
                terminal_step = t + 1
        tau = t - n + 1
        if tau >= 0:
            i_0, i_last = tau + 1, min(tau + n, terminal_step)
            _return = sum(reward_seq[i] for i in range(i_0, i_last + 1))
            if tau + n < terminal_step:
                _return += values[state_seq[tau + n]]
            values[state_seq[tau]] += alpha * (_return - values[state_seq[tau]])
        t += 1
    return values


def rms_single_run(n, policy, alpha, episodes):

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