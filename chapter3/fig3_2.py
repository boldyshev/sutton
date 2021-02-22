import matplotlib.pyplot as plt
from matplotlib.table import Table

# states as 5x5 table coordinates
states = [(x, y) for x in range(5) for y in range(5)]

# values as dict with keys being the states
values = dict((state, 0) for state in states)

# actions as dict of coordinate change
actions = {'south': (0, 1), 'north': (0, -1), 'east': (1, 0), 'west': (-1, 0)}

# all actions have equal probabilities
policy = 0.25


def state_reward(state, action):
    """State transitions on action

    :param state: previous state
    :type state: tuple
    :param action: action
    :type action: tuple
    :return: new state and reward
    :rtype: tuple
    """
    x, y = state
    dx, dy = action
    # A -> A'
    if (x, y) == (0, 1):
        return (4, 1), 10
    # B -> B'
    if (x, y) == (0, 3):
        return (2, 3), 5

    # new state
    x1, y1 = x + dx, y + dy

    # out of bounds
    if x1 in (-1, 5) or y1 in (-1, 5):
        return (x, y), -1
    # normal case: inside grid, not A or B state
    else:
        return (x1, y1), 0


def update_value(state, discount=0.9):
    """Update value for particular state

    :param state: state for which we calculate value
    :type state: tuple
    :param discount: discount coefficient
    :type discount: float
    :return: None
    :rtype: None
    """
    _value = 0
    for action in actions.values():
        next_state, reward = state_reward(state, action)
        _value += policy * (reward + discount * values[next_state])
    values[state] = _value


# evaluation accuracy
theta = 1e-6

# Iterative policy evaluation (pseudocode implementation from page 75)
delta = 1e-5
while delta > theta:
    delta = 0
    for state in states:
        prev_value = values[state]
        update_value(state)
        delta = max(delta, abs(prev_value - values[state]))


def fill_table(values):
    """Fill the values in a table

    :param values: state values
    :type values: dict
    :return: None
    :rtype: None
    """
    fig, ax = plt.subplots()
    tb = Table(ax, bbox=[0, 0, 1, 1])
    for (x, y), val in values.items():
        tb.add_cell(x, y, 1/5, 1/5, text=round(val, 1), loc='center', facecolor='white')

    ax.add_table(tb)
    ax.set_axis_off()


if __name__ == '__main__':
    fill_table(values)
    plt.show()
