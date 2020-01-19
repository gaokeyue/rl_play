from BlackJack.player import Player
import pandas as pd
import random
from collections import defaultdict


class Learner(Player):
    def __init__(self, num_learning_rounds):
        super().__init__()
        self._Q = {}
        self._Q_times = {}
        self._last_state = None
        self._last_action = None
        self._learning_rate = .1
        self.num_learning_rounds = num_learning_rounds
        self._learning_rate_interval_num = 1
        interval = self.num_learning_rounds/self._learning_rate_interval_num
        intervals = [interval * (i+1) for i in range(self._learning_rate_interval_num)]
        self._learning_rate_logger = dict(zip(intervals, [0] * self._learning_rate_interval_num))
        self._learning_rate_step = self._learning_rate / self._learning_rate_interval_num
        self._discount = .998
        self._epsilon = .9
        self._learning = True
        self._hit = defaultdict(list)
        self._stay = defaultdict(list)

    def reset_hand(self):
        self._hand = []
        self._last_state = None
        self._last_action = None

    def get_action(self, state=None):
        if state in self._Q and random.random() < self._epsilon:
            # action = max(self._Q[state], key=self._Q[state].get)
            if self._Q[state]['hit'] >= self._Q[state]['stay']:
                action = 'hit'
                self._Q_times[state]['hit'] += 1
            else:
                action = 'stay'
                self._Q_times[state]['stay'] += 1
        else:
            action = random.choice(['hit', 'stay'])
            if state not in self._Q:
                self._Q_times[state] = {}
                self._Q_times[state][action] = 1
                if action == 'hit':
                    self._Q_times[state]['stay'] = 0
                else:
                    self._Q_times[state]['hit'] = 0
                self._Q[state] = {}
                self._Q[state]['hit'] = 0
                self._Q[state]['stay'] = 0
            else:
                self._Q_times[state][action] += 1

        self._last_state = state
        self._last_action = action

        return action

    def update(self, new_state=None, reward=None, game=None):
        if self._learning:
            old = self._Q[self._last_state][self._last_action]

            if new_state in self._Q:
                new = self._discount * self._Q[new_state][max(self._Q[new_state], key=self._Q[new_state].get)]
            else:
                new = 0

            # if game % (self.num_learning_rounds / self._learning_rate_interval_num) == 0:
            #     self._learning_rate_logger[game] += 1
            #     if self._learning_rate_logger[game] == 1:
            #         self._learning_rate -= self._learning_rate_step
            #         print(game, reward, self._learning_rate)
            self._learning_rate = 1/(1+self._Q_times[self._last_state][self._last_action])

            self._Q[self._last_state][self._last_action] \
                = (1-self._learning_rate)*old + self._learning_rate*(reward+new)

            # if 'hit' in self._Q[self._last_state]:
            #     self._hit[self._last_state].append(self._Q[self._last_state]['hit'])
            # if 'stay' in self._Q[self._last_state]:
            #     self._stay[self._last_state].append(self._Q[self._last_state]['stay'])
            if self._last_state == (17, 10):
                print(f'{self._last_state}, {self._last_action}, {new_state}, old={round(old,8)}, '
                      f'new={round(self._Q[self._last_state][self._last_action],8)}, reward={reward}, '
                      f'Q={self._Q[self._last_state]}')

    def get_optimal_strategy(self):
        df = pd.DataFrame(self._Q).transpose()
        df['policy'] = df.apply(lambda x: 'hit' if x['hit'] >= x['stay'] else 'stay', axis=1)
        return df
