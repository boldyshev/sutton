#!/usr/bin/env python3

"""Figure 4.3, page 84"""

import random
import copy
import matplotlib.pyplot as plt
import numpy as np
PROBABILITY_HEADS = 0.4
HEADS = True
TAILS = False
THETA = 1e-9
STATES = np.arange(101)
VALUES_HISTORY_STEPS = (1, 2, 3, 32)

values = np.zeros(101)
values[100] = 1
policy = np.zeros(101)


def expected_value(values, state, action):
    return PROBABILITY_HEADS * values[state + action] + (1 - PROBABILITY_HEADS) * values[state - action]


def value_iteration():
    values_history = list()
    iteration = 0
    delta = 1
    while delta >= THETA:
        if iteration in VALUES_HISTORY_STEPS:
            values_history.append(copy.deepcopy(values))
        delta = 0
        for state in STATES[1:100]:
            actions = np.arange(min(state, 100 - state) + 1)

            old_value = values[state]
            values[state] = max([expected_value(values, state, action) for action in actions])
            delta = max(delta, abs(old_value - values[state]))
        iteration += 1
    values_history.append(values)
    return values_history


def policy_improve():
    for state in STATES[1:100]:
        actions = np.arange(min(state, 100 - state) + 1)
        action_values = [expected_value(values, state, action) for action in actions]
        policy[state] = actions[np.argmax(np.round(action_values[1:], 5)) + 1]


def fig4_3():
    values_history = value_iteration()
    policy_improve()
    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    for sweep, _value in zip(VALUES_HISTORY_STEPS, values_history):
        plt.plot(_value, label='sweep {}'.format(sweep))
    plt.plot(values, label='sweep final')
    plt.xlabel('Capital')
    plt.ylabel('Value estimates')
    plt.legend(loc='best')

    plt.subplot(2, 1, 2)
    plt.bar(STATES, policy)
    plt.xlabel('Capital')
    plt.ylabel('Final policy (stake)')

    plt.show()


if __name__ == '__main__':
    fig4_3()
