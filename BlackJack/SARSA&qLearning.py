import numpy as np
from _collections import defaultdict
import random
from game.blackjack import BlackJack
from tqdm import tqdm


class QLearning:

    def __init__(self, game):
        self.game = game
        self._alpha = 0.01
        self._gamma = 1
        self._epsilon = 0.1
        self.q = game.q_initializer()

    def print_policy(self):
        for key, value in self.q.items():
            if key[0] == 'K':
                print((key, value))

    def epsilon_greedy_action(self, state):
        greedy_action = max(self.q[state].items(), key=lambda x: x[1])[0]
        candidate_lst = list(self.q[state].keys())
        candidate_lst.append(greedy_action)
        probability_distribution = list(len(self.q[state]) * [self._epsilon]).append\
            (len(self.q[state]) * (1 - self._epsilon))
        next_step = str(*np.random.choice(candidate_lst, 1, probability_distribution))
        return next_step

    def fit(self):
        self.game.q_initializer()

        for i in tqdm(range(5 * 10 ** 5)):

            self.game.reset()
            is_terminal = False
            if i % 100 == 0:
                self._alpha *= 0.999
                # print(self.q[('K', 16, False)])
            while not is_terminal:
                current_state = self.game._state
                action = self.epsilon_greedy_action(current_state)
                next_state, reward, is_terminal = self.game.one_move(action)
                self.q[current_state][action] += \
                    self._alpha * (reward + self._gamma * max(self.q[next_state].values())
                                   - self.q[current_state][action])
                self.game._state = next_state


class SARSA():
    def __init__(self, game):
        self.game = game
        self._alpha = 0.01
        self._gamma = 1
        self._epsilon = 0.1
        self.q = game.q_initializer()

    def print_policy(self):
        for key, value in self.q.items():
            if key[0] == 'K':
                print((key, value))

    def epsilon_greedy_action(self, state):
        greedy_action = max(self.q[state].items(), key=lambda x: x[1])[0]
        candidate_lst = list(self.q[state].keys())
        candidate_lst.append(greedy_action)
        probability_distribution = list(len(self.q[state]) * [self._epsilon]).append \
            (len(self.q[state]) * (1 - self._epsilon))
        next_step = str(*np.random.choice(candidate_lst, 1, probability_distribution))
        return next_step

    def fit(self):

        for _ in tqdm(range(5 * 10 ** 5)):
            is_terminal = False
            self.game.reset()
            next_action = self.epsilon_greedy_action(self.game._state)
            while not is_terminal:
                current_state = self.game._state
                action = next_action
                next_state, reward, is_terminal = self.game.one_move(action)
                next_action = self.epsilon_greedy_action(next_state)
                self.q[current_state][action] += \
                    self._alpha * (reward + self._gamma * self.q[next_state][next_action]
                                   - self.q[current_state][action])
                self.game._state = next_state

    def epsilon_greedy_action(self, state):
        greedy_action = max(self.q[state].items(), key=lambda x: x[1])[0]
        candidate_lst = list(self.q[state].keys())
        candidate_lst.append(greedy_action)
        probability_distribution = list(len(self.q[state]) * [self._epsilon]).append\
            (len(self.q[state]) * (1 - self._epsilon))
        next_step = str(*np.random.choice(candidate_lst, 1, probability_distribution))
        return next_step


if __name__ == '__main__':
    q = QLearning(BlackJack())
    q.fit()
    q.print_policy()
    # sarsa = SARSA(BlackJack())
    # sarsa.fit()
    # sarsa.print_policy()






