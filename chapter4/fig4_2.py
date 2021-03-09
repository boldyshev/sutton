import time
import multiprocessing as mp
from math import exp, factorial, pow

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# maximum car number at each location
MAX_CARS = 20
# maximal number of cars to be moved overnight
MAX_MOVE = 5
# policy evaluation accuracy
THETA = 1e-2

# all possible states
states = [(x, y) for x in range(MAX_CARS + 1) for y in range(MAX_CARS + 1)]
# values as dictionary with states being the keys
values = dict((state, 0) for state in states)
# same for policy
policy = dict((state, 0) for state in states)


def poisson_prob(k, mu):
    """The probability of k occurrences of a random variable having poisson distribution with expected value mu

    :param k: number of occurrences
    :type k: int
    :param mu: expected value
    :type mu: int
    :return: probability of k
    :rtype: float
    """
    return exp(-mu) * pow(mu, k) / factorial(k)


# all possible pairs of car requests with the probability higher than 0.01 for the first and second location
# say, the probability to get a request for 8 cars at the first location (expected value is 3) is ~0.0081
car_requests = [(i, j) for i in range(8) for j in range(10)]
# the same for car returns
car_returns = [(i, j) for i in range(8) for j in range(7)]
# all possible pairs of requests and returns
requests_returns = [(i, j) for i in car_requests for j in car_returns]

# precalculate probabilities for all possible combinations of requests and returns in two locations
poisson_prob_product = dict()
for car_request, car_return in requests_returns:
    # expectations for requested number of cars are 3 and 4 for the first and he second location respectively
    prob_request = poisson_prob(car_request[0], 3) * poisson_prob(car_request[1], 4)
    # expectations for returned number of cars are 3 and 2 for the first and he second location respectively
    prob_return = poisson_prob(car_return[0], 3) * poisson_prob(car_return[1], 2)
    poisson_prob_product[car_request, car_return] = prob_request * prob_return


def state_update(state, car_request, car_return):
    """State change given the number of requested and returned cars and the prize for renting cars

    :param state: state before request and return
    :type state: tuple
    :param car_request: number of cars requested at the first and the second location
    :type car_request: tuple
    :param car_return: number of cars returned at the first and the second location
    :type car_return: tuple
    :return: new state and reward for rented cars
    :rtype: tuple
    """
    # can't request more cars than available
    car_request = min(state[0], car_request[0]), min(state[1], car_request[1])

    # cars on both locations after request
    next_state = [state[0] - car_request[0], state[1] - car_request[1]]

    # get $10 for each rented car
    reward = sum(car_request) * 10

    # get returned cars, remove if their number exceeds 20
    next_state[0] = min(next_state[0] + car_return[0], 20)
    next_state[1] = min(next_state[1] + car_return[1], 20)

    return tuple(next_state), reward


def expected_value(values, state, action, discount=0.9):
    """Calculate expected values for a given action

    :param values: all known values so far
    :type values: dict
    :param state: state which value is updated
    :type state: tuple
    :param action: action taken in this state
    :type action: int
    :param discount: discount
    :type discount: float
    :return: new value
    :rtype: float
    """
    # move cars overnight
    state = min(state[0] - action, 20), min(state[1] + action, 20)
    # pay for car movement
    new_value = -2 * abs(action)

    for car_request, car_return in requests_returns:
        prob = poisson_prob_product[car_request, car_return]
        next_state, reward = state_update(state, car_request, car_return)
        new_value += prob * (reward + discount * values[next_state])
    return new_value


def eval_state(values, state, action):
    """Function for parallel policy evaluation

    :param values: all known values so far
    :type values: dict
    :param state: state which value is updated
    :type state: tuple
    :param action: action taken in this state
    :type action: int
    :return: state, new value and its difference with the previous value
    :rtype: tuple
    """
    new_value = expected_value(values, state, action)
    delta = abs(values[state] - new_value)

    return state, new_value, delta


def policy_evaluation():
    """Parallel implementation of policy evaluation algorithm from RL book page 80

    :return: None
    """

    delta = 1
    while delta > THETA:
        delta = 0
        with mp.Pool(mp.cpu_count()) as pool:
            result = pool.starmap(eval_state, [(values, state, policy[state]) for state in states])
        for state, new_value, delta1 in result:
            values[state] = new_value
            delta = max(delta, delta1)

        print('\r', f'policy evaluation {delta}'[:27], f'-> {THETA}', end='')


def argmax(iterable):
    """Returns the index of the maximum element for python built-in iterables (e.g. lists or tuples).
    Turns out to be faster than numpy.argmax on low-dimensional vectors.

    :param iterable iterable: The vector in which to find the index of the maximum element
    :return: Maximum element index
    :rtype: Int
    """
    return max(range(len(iterable)), key=lambda x: iterable[x])


def improve_step(state):
    """Function for parallel policy improvement

    :param state: state for which we improve policy
    :return: same state, best action with respect to the expected value and if it matches the current policy
    :rtype: tuple
    """
    # can't move overnight more cars, than there are at the location
    actions = range(-min(state[1], 5), min(state[0], 5) + 1)
    # list of values of all possible action
    action_values = [expected_value(values, state, a) for a in actions]
    # action with maximal value
    optimal_action = actions[argmax(action_values)]
    # if the current policy chooses maximal value action
    is_optimal = optimal_action == policy[state]

    return state, optimal_action, is_optimal


def policy_improvement():
    """Parallel implementation of policy improvement algorithm from RL book page 80

    :return: whether the policy is stable
    :rtype: bool
    """
    with mp.Pool(mp.cpu_count()) as pool:
        result = pool.map(improve_step, states)
    stables = []
    for state, optimal_action, is_optimal in result:
        policy[state] = optimal_action
        stables.append(is_optimal)
    policy_stable = False not in stables
    return policy_stable


def heatmap(data, title, axes, iteration):
    """Plot the heatmap for policy or values

    :param data:
    :param title:
    :param axes:
    :param iteration:
    :return:
    """
    data_arr = np.empty((MAX_CARS + 1, MAX_CARS + 1))
    for state, value in data.items():
        x, y = state
        data_arr[x][y] = value
    h = sns.heatmap(data_arr, ax=axes[iteration])
    h.set_ylabel('#Cars at first location')
    h.set_xlabel('#Cars at second location')
    h.set_title(title)
    h.invert_yaxis()
    plt.rc('mathtext', fontset="cm")


def policy_iteration():
    """Policy iteration implemention from RL book page 80

    :return: None
    """

    policy_stable = False
    iteration = 0
    fig, axes = plt.subplots(2, 3, figsize=(40, 20))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    axes = axes.flatten()
    while not policy_stable:
        policy_title = r'$\pi_{}$'.format(iteration)
        heatmap(policy, policy_title, axes, iteration)
        print('\niteration ', iteration)
        t0 = time.perf_counter()
        policy_evaluation()
        t1 = time.perf_counter()
        print(f' done in {round(t1 - t0, 3)} sec')

        print(' policy improvement...', end=' ')
        policy_stable = policy_improvement()
        t2 = time.perf_counter()
        print(f' done in {round(t2 - t1, 3)} sec')

        iteration += 1
    value_title = r'$v_{\pi_4}$'
    heatmap(values, value_title, axes, iteration)


if __name__ == '__main__':
    policy_iteration()
    plt.show()
