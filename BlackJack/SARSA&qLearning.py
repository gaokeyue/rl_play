import numpy as np
from _collections import defaultdict
import random
from game.Blackjack import blackjack
from tqdm import tqdm


class QLearning:
    """
    Dynamics needs atr: state, init_state, if_terminal, q
                   func: next_state, sys_init
    """

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
    def print_policy(self):
        self.state = (0, 0)
        print(self.state)
        iter = 0
        while (self.state != self._terminal_state) & (iter < 100):
            iter += 1
            state = self.state
            action = max(self.q[state].items(), key=lambda x: x[1])[0]
            next_state, _ = self.next_state(action)
            self.state = next_state
            print(next_state)

    def fit(self):
        self.sys_init()
        for i in range(10 ** 8):
            self.state = (0, 0)
            next_action = self.epsilon_greedy_action(self.state)
            if i % 1 == 0:
                print(f'{i} iterations now')
            while self.state != self._terminal_state:
                current_state = self.state
                action = next_action
                next_state, reward = self.next_state(action)
                next_action = self.epsilon_greedy_action(next_state)
                self.q[current_state][action] += \
                    self._alpha * (reward + self._gamma * self.q[next_state][next_action]
                                   - self.q[current_state][action])
                self.state = next_state
        self.print_policy()

    def epsilon_greedy_action(self, state):
        greedy_action = max(self.q[state].items(), key=lambda x: x[1])[0]
        candidate_lst = list(self.q[state].keys())
        candidate_lst.append(greedy_action)
        probability_distribution = list(len(self.q[state]) * [self._epsilon]).append\
            (len(self.q[state]) * (1 - self._epsilon))
        next_step = str(*np.random.choice(candidate_lst, 1, probability_distribution))
        return next_step


if __name__ == '__main__':
    # cliff_walking.q_learning()
    # cliff_walking.sarsa()
    q = QLearning(blackjack)
    q.fit()
    q.print_policy()
    # print(q._name)






