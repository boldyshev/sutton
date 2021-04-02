import matplotlib.pyplot as plt

from fig3_2 import state_reward, fill_table

# states as 5x5 table coordinates
states = [(x, y) for x in range(5) for y in range(5)]

# values as dict with keys being the states
values = dict((state, 0) for state in states)

# actions as dict of coordinate change
actions = {'south': (0, 1), 'north': (0, -1), 'east': (1, 0), 'west': (-1, 0)}


def update_value_optimal(state, discount=0.9):
    """Optimal value calculation

    :param state: state for which we calculate value
    :type state: tuple
    :param discount: discount coefficient
    :type discount: float
    :return: None
    :rtype: None
    """
    _value = []
    for action in actions.values():
        next_state, reward = state_reward(state, action)
        _value.append(reward + discount * values[next_state])
    values[state] = max(_value)


# evaluation accuracy
theta = 1e-6

# difference between values calculated on sequential steps
delta = 1e-5
while delta > theta:
    delta = 0
    for state in states:
        prev_value = values[state]
        update_value_optimal(state)
        delta = max(delta, abs(prev_value - values[state]))

fill_table(values)
plt.show()
