from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.table import Table

# states are cells of 4x4 grid
states = [(x, y) for x in range(4) for y in range(4)]
# actions
actions = {'north': (0, -1), 'south': (0, 1), 'east': (1, 0), 'west': (-1, 0)}


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
    if (x, y) in [(0, 0), (3, 3)]:
        return (x, y), 0
    # new state
    x1, y1 = x + dx, y + dy
    if x1 in (-1, 4) or y1 in (-1, 4):
        return (x, y), -1
    else:
        return (x1, y1), -1


def update_value_out(values, state, discount=0.9):
    """Update value for particular state

    :param values: values
    :type values: dict
    :param state: state for which we calculate value
    :type state: tuple
    :param discount: discount coefficient
    :type discount: float
    :return: new value
    :rtype: float
    """
    if state in [(0, 0), (3, 3)]:
        return 0
    _value = 0
    for action in actions.values():
        next_state, reward = state_reward(state, action)
        _value += policy * (reward + discount * values[next_state])
    return _value


k_values = (0, 1, 2, 3, 10, 1000)
fig, axs = plt.subplots(len(k_values), figsize=(4, 16))

for i, k in enumerate(k_values):
    values = dict((state, 0) for state in states)
    values[(0, 0)] = 0
    values[(3, 3)] = 0
    policy = 0.25

    for _ in range(k):
        # Two-array version of the iterative policy evaluation thanks to ShangtongZhang's repo
        # https://github.com/ShangtongZhang/reinforcement-learning-an-introduction
        old_values = deepcopy(values)
        for state in states:
            prev_value = values[state]
            values[state] = update_value_out(old_values, state, discount=1)
    tb = Table(axs[i], bbox=[0, 0, 1, 1])
    for (x, y), val in values.items():
        tb.add_cell(x, y, 1 / 4, 1 / 4, text=round(val, 1),
                    loc='center', facecolor='white')

    axs[i].title.set_text(f'k = {k}')
    axs[i].add_table(tb)
    axs[i].set_axis_off()

plt.show()
