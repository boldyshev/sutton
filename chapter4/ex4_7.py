#!/usr/bin/env python3

"""Exercise 4.7, page 81"""

import copy
import math
import time
import itertools

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# maximum car number at each location
MAX_CARS = 20

# all possible states -- cartesian product of 21 (as 0 is a possible state)
STATES = tuple(itertools.product(range(MAX_CARS + 1), repeat=2))

# maximal number of cars to move overnight
MAX_MOVE = 5

# policy evaluation accuracy
THETA = 1e-2

# initialise values and policy
# values as 2d array axis0 and axis1 (x, y) being the numbers of cars at first and second locations respectively
values = np.zeros((MAX_CARS + 1,) * 2)

# action chosen at each state (integers from -5 to 5)
policy = np.zeros((MAX_CARS + 1,) * 2, dtype=int)

# rewards and transition probabilities will be precalculated
# 4d array for cumulative reward for transition from one state (x, y) to another (x1, y1)
rewards = np.zeros((MAX_CARS + 1, ) * 4)

# 4d array for transition probabilities between all states
transition_probs = np.zeros((MAX_CARS + 1, ) * 4)


def poisson_prob(n, _lambda):
    """The probability of k occurrences of a random variable having poisson distribution with expected value mu

    :param n: number of occurrences
    :type n: int
    :param _lambda: expected value
    :type _lambda: int
    :return: probability of k
    :rtype: float
    """
    return math.exp(-_lambda) * pow(_lambda, n) / math.factorial(n)


def precalculate():
    """Precalculate rewards and transition probabilities

    :return: rewards and transition probabilities
    """

    # precalculate lists of probability mass functions for different car numbers
    # see fig4_2.ipynb for clarification
    poisson_probs = dict()
    for _lambda in (2, 3, 4):
        for cars_num in range(MAX_CARS + 1):
            poisson_probs[cars_num, _lambda] = [poisson_prob(k, _lambda) for k in range(cars_num + 1)]
            poisson_probs[cars_num, _lambda][-1] += 1 - sum(poisson_probs[cars_num, _lambda])

    t0 = time.perf_counter()

    # x and y state for numbers of cars at first and second locations respectively
    for x, y in STATES:
        print('\r', f'Precalculating rewards and transition probabilities for state {x, y}', end=' ')
        # can't request more cars then available at either location
        for request_x, req_quest in itertools.product(range(x + 1), range(y + 1)):
            # cars left after request
            x1, y1 = x - request_x, y - req_quest

            # probability for the number of requested cars at both locations
            prob_req = poisson_probs[x, 3][request_x] * poisson_probs[y, 4][req_quest]
            # $10 for each car rented

            reward = (request_x + req_quest) * 10
            # can't return more than 20 minus number of cars left (cars over 20 disappear from the problem)
            for return_x, return_y in itertools.product(range(21 - x1), range(21 - y1)):
                # cars on both locations after return
                x2, y2 = x1 + return_x, y1 + return_y

                # probability for the number of cars returned to both locations
                prob_ret = poisson_probs[20 - x1, 3][return_x] * poisson_probs[20 - y1, 2][return_y]

                # probability of this particular transition from (x, y) to (x2, y2)
                prob_product = prob_req * prob_ret

                # update reward for (x, y) to (x2, y2) transition
                rewards[x, y, x2, y2] += reward * prob_product

                # update probability of (x, y) to (x2, y2) transition
                transition_probs[x, y, x2, y2] += prob_product

    t1 = time.perf_counter()
    print('done in ', round(t1 - t0, 3), 'seconds')


def expected_value(values, state, action, discount=0.9):
    """Calculate expected value for a given action

    :param state: state which value is updated
    :type state: tuple
    :param action: action taken in this state
    :type action: int
    :param discount: discount
    :type discount: float
    :return: new value
    :rtype: float
    """
    x, y = state

    # move cars overnight, actions are validated during the policy improvement
    x, y = min(x - action, 20), min(y + action, 20)

    if action < 0:
        action += 1
    # pay $2 for each moved car
    new_value = -2 * abs(action)

    if x > 10:
        new_value -= 4
    if y > 10:
        new_value -= 4

    # Bellman equation
    new_value += np.sum(rewards[x, y]) + np.sum(transition_probs[x, y] * values) * discount

    return new_value


def argmax(iterable):
    """Returns the index of the maximum element for python built-in iterables (e.g. lists or tuples).
    Turns out to be faster than numpy.argmax on low-dimensional vectors.

    :param iterable iterable: The vector in which to find the index of the maximum element
    :return: Maximum element index
    :rtype: Int
    """
    return max(range(len(iterable)), key=lambda x: iterable[x])


def policy_evaluation():
    """Policy evaluation implementation (step 2 of algorithm at page 80)

    :return: None
    """
    delta = 1
    while delta > THETA:
        delta = 0
        old_values = copy.deepcopy(values)
        for state in STATES:
            values[state] = expected_value(old_values, state, policy[state])
            delta = max(delta, abs(values[state] - old_values[state]))


def policy_improvement():
    """Policy improvement implementation (step 3 of algorithm at page 80)

    :return: None
    """
    policy_stable = True
    for state in STATES:
        old_action = policy[state]

        # get all possible actions
        x, y = state
        actions = range(-min(y, MAX_MOVE), min(x, MAX_MOVE) + 1)

        # list of action values
        action_values = [expected_value(values, state, action) for action in actions]

        # optimal action
        policy[x, y] = actions[argmax(action_values)]

        if old_action != policy[state]:
            policy_stable = False

    return policy_stable


def heatmap(data, title, axes, iteration):
    """Plot the heatmap for policy or values

    :param data:
    :param title:
    :param axes:
    :param iteration:
    :return:
    """
    h = sns.heatmap(data, ax=axes[iteration])
    h.set_ylabel('#Cars at first location')
    h.set_xlabel('#Cars at second location')
    h.set_title(title)
    h.invert_yaxis()
    plt.rc('mathtext', fontset="cm")


def policy_iteration():
    """Policy iteration implementation from RL book page 80

    :return: None
    """
    precalculate()

    # plotting
    fig, axes = plt.subplots(2, 3, figsize=(40, 20))
    # plt.subplots_adjust(wspace=0.1, hspace=0.2)
    axes = axes.flatten()

    iteration = 0
    policy_stable = False
    while not policy_stable:
        print('\r', 'policy iteration ', iteration, end='')
        # plot policy
        policy_title = r'$\pi_{}$'.format(iteration)
        heatmap(policy, policy_title, axes, iteration)

        # step2
        policy_evaluation()

        # step3
        policy_stable = policy_improvement()

        iteration += 1

    # plot values
    value_title = r'$v_{\pi_4}$'
    heatmap(values, value_title, axes, iteration)


if __name__ == '__main__':
    policy_iteration()
    plt.show()
