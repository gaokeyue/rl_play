import numpy as np
from _collections import defaultdict
import random
from MC import cards, cards_count


class Dynamics:
    def __init__(self):
        self._name = 'CliffWalking'
        self._space_size = (4, 12)
        self._terminal_state = (0, 11)
        self._cliff = list(range(1, 11))
        self.q = defaultdict(dict)
        self.init_state = (0, 0)
        self.state = (0, 0)

    def get_action(self, state):
        action_lst = []
        if state[0]:
            action_lst.append('down')
        if state[1]:
            action_lst.append('left')
        if state[0] != self._space_size[0] - 1:
            action_lst.append('up')
        if state[1] != self._space_size[1] - 1:
            action_lst.append('right')
        return action_lst

    def take_action(self, action):
        if action == 'left':
            next_state = (self.state[0], self.state[1] - 1)
        elif action == 'down':
            next_state = (self.state[0] - 1, self.state[1])
        elif action == 'right':
            next_state = (self.state[0], self.state[1] + 1)
        elif action == 'up':
            next_state = (self.state[0] + 1, self.state[1])
        else:
            print(f'Error: {action} is not a formal action')
        return next_state

    def sys_init(self):
        for x in range(self._space_size[0]):
            for y in range(self._space_size[1]):
                temp_state = (x, y)
                actions = self.get_action(temp_state)
                # if temp_state == self._terminal_state:
                #     for action in actions:
                #         self.q[temp_state][action] = 0
                #         self.pai[temp_state][action] = 1
                # else:
                for action in actions:
                    self.q[temp_state][action] = -(self._space_size[1] - y - 1 + x) / 10

    def next_state(self, action):
        next_state = self.take_action(action)
        if (not next_state[0]) & (next_state[1] in self._cliff):
            next_state = (0, 0)
            reward = -100
        else:
            reward = -1
        return next_state, reward

    @property
    def if_terminal(self):

        return True if self.state == self._terminal_state else False


class BlackJack:

    def __init__(self):
        self.q = defaultdict(dict)
        self.state = ('shown_card', 'hands_sum', 'usable_or_not')
        self._actions = ['hit', 'stand']
        self.if_terminal = False

    def init_state(self):
        # usable_ace = False
        # shown_card = random.choice(cards)
        shown_card = 'K'
        hands_sum = random.randint(16, 21)
        # hands_sum = 16
        # card_one, card_two = random.choice
        # s(cards, k=2)
        # hands_sum = cards_count[card_one] + cards_count[card_two]
        # if card_two == 'A' or card_two == 'A':
        #     usable_ace = True
        # if hands_sum == 22:
        #     hands_sum = 12
        # usable_ace = random.choice([True, False])
        usable_ace = False
        return shown_card, hands_sum, usable_ace

    def dealer(self, seed=None):
        random.seed = seed
        usable_ace = False
        card_two = random.choice(cards)
        shown_card = self.state[0]
        count = cards_count[shown_card] + cards_count[card_two]
        if card_two == 'A' or shown_card == 'A':
            usable_ace = True
        if count == 22:
            count = 12
        while count < 17:
            hit_card = random.choice(cards)
            count += cards_count[hit_card]
            if count > 21:
                if (not usable_ace) and (hit_card != 'A'):
                    return 0
                else:
                    count = count - 10
                    usable_ace = False
            if (hit_card == 'A') and usable_ace:
                usable_ace = True
        return count

    def sys_init(self):
        self.q['busted']['busted'] = 0
        self.q['stand']['stand'] = 0
        for shown_card in cards:
            for hands_sum in range(12, 22):
                state = (shown_card, hands_sum, True)
                self.q[state]['hit'] = 0
                self.q[state]['stand'] = 0
                state = (shown_card, hands_sum, False)
                self.q[state]['hit'] = 0
                self.q[state]['stand'] = 0

    def next_state(self, action):
        if action == 'hit':
            new_card = random.choice(cards)
            shown_card, hands_sum, usable_ace = self.state
            hands_sum += cards_count[new_card]
            new_state = (shown_card, hands_sum, usable_ace)
            self.state = new_state
            if hands_sum > 21:

                if (not usable_ace) & (new_card is not 'A'):
                    reward = -1
                    self.if_terminal = True
                    new_state = "busted"
                else:
                    if hands_sum == 32:
                        reward = -1
                        self.if_terminal = True
                        new_state = "busted"
                    else:
                        hands_sum -= 10
                        if (new_card is 'A') & usable_ace:
                            usable_ace = True
                        else:
                            usable_ace = False
                        new_state = (shown_card, hands_sum, usable_ace)
                        reward = 0
            else:
                reward = 0
        else:
            self.if_terminal = True
            dealer_pts = self.dealer()
            if self.state[1] < dealer_pts:
                reward = -1
            elif self.state[1] == dealer_pts:
                reward = 0
            else:
                reward = 1
            new_state = 'stand'
        return new_state, reward


class QLearning(BlackJack):
    """
    Dynamics needs atr: state, init_state, if_terminal, q
                   func: next_state, sys_init
    """

    def __init__(self):
        BlackJack.__init__(self)
        self._alpha = 0.01
        self._gamma = 1
        self._epsilon = 0.1

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
        self.sys_init()
        for i in range(5 * 10 ** 5):

            self.state = self.init_state()
            self.if_terminal = False
            if i % 100 == 0:
                self._alpha *= 0.999
                print(f'{i} iterations now')
                print(self.q[('K', 16, False)])
            while not self.if_terminal:
                current_state = self.state
                action = self.epsilon_greedy_action(current_state)
                next_state, reward = self.next_state(action)
                self.q[current_state][action] += \
                    self._alpha * (reward + self._gamma * max(self.q[next_state].values())
                                   - self.q[current_state][action])
                self.state = next_state


class SARSA(Dynamics):
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
    q = QLearning()
    q.fit()
    q.print_policy()
    # print(q._name)






