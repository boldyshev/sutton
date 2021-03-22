#!/usr/bin/env python3

"""Figure 4.3, page 84"""

import copy

import numpy as np
import matplotlib.pyplot as plt

# probability to win the stake
PROBABILITY_HEADS = 0.25

# states with $0 or $100 are not considered as the are terminal
STATES = np.arange(1, 100)

# accuracy
THETA = 1e-9

values = np.zeros(101)
values[100] = 1
policy = np.zeros(101)


def actions(state):
    """Get possible actions for a given state

    :param state: State
    :type state: int
    :return: Possible actions
    :rtype: numpy array
    """
    return np.arange(1, min(state, 100 - state) + 1)


def expected_update(values, state, action):
    """Compute expected value for given state and action

    :param values: Current values
    :type: numpy array
    :param state: State
    :type: int
    :param action: Action
    :type: int
    :return: expected value
    :rtype: float
    """
    return PROBABILITY_HEADS * values[state + action] + (1 - PROBABILITY_HEADS) * values[state - action]


def policy_evaluate():
    """Policy evaluation

    :return: Difference between old and new value
    :rtype: float
    """
    delta = 0
    for state in STATES:
        old_value = values[state]
        values[state] = max([expected_update(values, state, action) for action in actions(state)])
        delta = max(delta, abs(old_value - values[state]))
    return delta


def policy_improve(round_digits=4):
    """Policy improvement

    :param round_digits: Precision of expected value
    :type round_digits: int
    :return: None
    :rtype: None
    """
    for state in STATES:
        _actions = actions(state)
        action_values = [round(expected_update(values, state, action), round_digits) for action in _actions]
        optimal_action = _actions[np.argmax(action_values)]
        policy[state] = optimal_action


def value_iteration():
    """Value iteration algorithm implementation (page 83)

    :return: Values after each sweep of policy evaluation
    :rtype: list
    """
    values_history = [values]

    sweep = 0
    delta = 1
    while delta >= THETA:
        delta = policy_evaluate()
        sweep += 1
        values_history.append(copy.deepcopy(values))
    return values_history


def plot(values_history, policy):
    """Plot values and policy

    :param values_history: Values after each sweep of policy evaluation
    :type values_history: list
    :param policy: Policy
    :type policy: numpy array
    :return: None
    :rtype: None
    """
    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)

    values_history_sweeps = (1, 2, 3, len(values_history)//2)
    for sweep in values_history_sweeps:
        plt.plot(values_history[sweep], label=f'sweep {sweep}')
    plt.plot(values, label='sweep final')

    plt.xlabel('Capital')
    plt.ylabel('Value estimates')
    plt.legend(loc='best')

    plt.subplot(2, 1, 2)
    plt.bar(np.arange(len(policy)), policy)
    plt.xlabel('Capital')
    plt.ylabel('Final policy (stake)')

    plt.show()


def fig4_3():
    values_history = value_iteration()
    policy_improve()
    plot(values_history, policy)


if __name__ == '__main__':
    fig4_3()
