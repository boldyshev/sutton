import random
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


class Bandit:

    def __init__(self,
                 n_arms=10,
                 true_value=0,
                 estim_value=0,
                 eps=0.1,
                 alpha=0.1):

        self.init_args = deepcopy(vars(self))

        self.q = list(np.random.normal(true_value, 1, size=n_arms))
        self.q_est = [estim_value] * n_arms

        self.action_counts = [0] * n_arms
        self.rewards = list()
        self.optimals = list()
        self.optimal = self.argmax(self.q)
        self.action = 0

    @staticmethod
    def argmax(iterable):
        """Returns the index of the maximum element for python built-in iterables (e.g. lists or tuples).
        Turns out to be faster than numpy.argmax on low-dimensional vectors.

        :param iterable iterable: The vector for which we want to find the index of the maximum element
        :return: Maximum element index
        :rtype: Int
        """
        return max(range(len(iterable)), key=lambda x: iterable[x])

    @staticmethod
    def percent(arr):
        """Optimal actions percentage

        :param arr: Optimal actions binary vector
        :type arr: numpy.array
        :return: Optimal action percentage in arr
        :rtype: float
        """

        return 100 * np.sum(arr, axis=0) / arr.shape[0]

    @staticmethod
    def plot(datay,
             labels,
             ylabel,
             datax=None,
             xlabel='Steps',
             colors=('blue', 'red', 'green', 'black'),
             fig_size=(18, 8),
             font_size=20):
        """Plotting average rewards or optimal actions

        :param datay: The set of Y-axis values to be plotted in one graph
        :type datay: iterable
        :param labels: Legend labels
        :type labels: iterable
        :param ylabel: Y-axis label
        :type ylabel: str
        :param datax: X-axis values
        :type datax: iterable
        :param xlabel: X-axis label
        :type xlabel: str
        :param colors: Plot colors
        :type colors: iterable
        :param fig_size: Figure size
        :type fig_size: tuple
        :param font_size: Font size
        :type font_size: int
        :return: Matplotlib axes object
        :rtype: matplotlib.axes
        """
        # create figure
        fig, ax = plt.subplots(figsize=fig_size)

        # plot graphs
        for i, arr in enumerate(datay):
            if not datax:
                x = range(len(arr))
            else:
                x = datax[i]
            ax.plot(x, arr, label=labels[i], color=colors[i])

        # labels etc.
        ax.set_ylabel(ylabel, fontsize=font_size)
        ax.set_xlabel(xlabel, fontsize=font_size)
        ax.tick_params(axis="x", labelsize=font_size)
        ax.tick_params(axis="y", labelsize=font_size)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.legend(loc='lower right', fontsize=font_size)
        if ylabel == '% Optimal action':
            ax.yaxis.set_major_formatter(PercentFormatter())
            ax.set_ylim(0, 100)
        plt.rc('mathtext', fontset="cm")
        return ax

    def one_step(self):
        self.choose_action()
        self.optimals.append(self.action is self.optimal)
        reward = np.random.normal(self.q[self.action], 1)
        self.rewards.append(reward)
        self.update_estimation()

    def one_step_nonstationary(self):
        self.optimal = self.argmax(self.q)
        self.one_step()
        self.q += np.random.normal(0, 0.01, size=10)

    def stationary(self, steps):
        self.__init__(**self.init_args)
        for i in range(steps):
            self.one_step()
        return self.rewards, self.optimals

    def nonstationary(self, steps):
        self.__init__(**self.init_args)
        for i in range(steps):
            self.one_step_nonstationary()
        return self.rewards, self.optimals


class EpsGreedyConstant(Bandit):

    def choose_action(self):
        if random.random() < 1 - self.init_args['eps']:
            self.action = self.argmax(self.q_est)
        else:
            self.action = random.randint(0, 9)

    def update_estimation(self):
        self.q_est[self.action] += (self.rewards[-1] - self.q_est[self.action]) * self.init_args['alpha']


class EpsGreedy(EpsGreedyConstant):

    def choose_action(self):
        super().choose_action()
        self.action_counts[self.action] += 1

    def update_estimation(self):
        self.q_est[self.action] += (self.rewards[-1] - self.q_est[self.action]) / self.action_counts[self.action]
