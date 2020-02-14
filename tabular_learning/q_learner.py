from tabular_learning import Agent
import numpy as np
from tqdm import tqdm
import pandas as pd
from collections import defaultdict


class QLearner(Agent):
    """q_learning, sarsa, expected sarsa"""

    def __init__(self, game, epsilon=0.1, alpha=0.1, gamma=1):

        self.game = game
        self.epsilon = epsilon
        self.alpha = alpha

    def q_initializer(self):
        """This method is not an abstract method because some games are of the 'after state' type, hence
        should be initialized as value function, e.g. defaultdict(int).
        """
        if self.game.state_space is None:
            return defaultdict(lambda: defaultdict(int))
        q = {}
        for state in self.game.state_space:
            actions = self.game.available_actions(state)
            action_value = {action: 0 for action in actions}
            if len(action_value) == 0:  # actions is empty, i.e. state is terminal
                action_value[None] = 0
            q[state] = action_value
        return q

    def choose_action(self, state, q, random=True):
        if np.random.random() < self.epsilon:
            chosen_action = np.random.choice(self.game.available_actions(state))
        else:
            temp_action = self.greedy_pick(q[state], thresh=10**-7)
            if len(temp_action)==1:
                chosen_action = temp_action[0]
            else:
                chosen_action = np.random.choice(temp_action)
        return chosen_action

    def q(self, episodes=50 * 10 ** 5):
        game = self.game
        q = self.q_initializer()
        for i in tqdm(range(episodes)):
            state = game.reset()
            is_terminal = False
            while not is_terminal:
                action = self.choose_action(state, q)
                next_state, reward, is_terminal = game.one_move(action)
                target = max(q[next_state].values())
                q[state][action] += self.alpha * (reward + game.gamma * target - q[state][action])
                state = next_state
        return q

    def sarsa(self, episodes=50 * 10 ** 5):
        game = self.game
        q = self.q_initializer()
        for i in tqdm(range(episodes)):
            state = game.reset()
            is_terminal = False
            action = self.choose_action(state, q)
            while not is_terminal:
                next_state, reward, is_terminal = game.one_move(action)
                next_action = self.choose_action(next_state, q)
                q[state][action] += self.alpha * (reward + game.gamma
                                                          * q[next_state][next_action] - q[state][action])
                state = next_state
                action = next_action
        return q

    def expected_sarsa(self, episodes):
        game = self.game
        q = self.q_initializer()
        for i in tqdm(range(episodes)):
            state = game.reset()
            is_terminal = False
            reward = 0
            action = self.choose_action(state, q)
            while not is_terminal:
                next_state, reward, is_terminal = game.one_move(action)
                next_action = self.choose_action(next_state, q, random=False)
                candidate_lst = list(v for v in q[next_state].values()) + [q[next_state][next_action]]
                prob_dist = len(q[next_state]) * [self.epsilon] + \
                            [(len(q[next_state]) * (1 - self.epsilon))]
                prob_list = [prob / sum(prob_dist) for prob in prob_dist]
                target = np.dot(candidate_lst, prob_list)
                q[state][action] += self.alpha * (reward + game.gamma * target - q[state][action])
                state = next_state
                action = next_action
        return q


if __name__ == '__main__':
    from game.blackjack import BlackJack

    test = BlackJack()
    test_q = QLearner(test, 0.1, 0.0005, 1)
    q1 = test_q.q(5 * 10 ** 5)

