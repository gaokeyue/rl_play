from tabular_learning import Agent
import numpy as np
from tqdm import tqdm


class QLearner(Agent):
    """q_learning, sarsa, expected sarsa"""

    def __init__(self, game, epsilon=0.1, alpha=0.01, gamma=1):

        self.game = game
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def choose_action(self, state, q, random = True):
        if np.random.binomial(1, self.epsilon) == 1 & random:
            chosen_action = np.random.choice(self.game.available_actions(state))
        else:
            chosen_action = np.random.choice(self.greedy_pick(q[state], find_one=False))
        return chosen_action

    def q(self, episodes = 50*10**5):
        game = self.game
        q = game.q_initializer()
        for i in tqdm(range(episodes)):
            state = game.reset()
            is_terminal = False
            reward = 0
            while not is_terminal:
                # state = game._state
                action = self.choose_action(state, q)
                next_state, reward, is_terminal = game.one_move(action)
                target = np.max(list(value.average for key, value in q[next_state].items()))
                q[state][action].average += self.alpha * (reward +\
                                self.gamma * target - q[state][action].average)
                state = next_state
        return q


    def sarsa(self, episodes = 50*10**5):
        game = self.game
        q = game.q_initializer()
        for i in tqdm(range(episodes)):
            state = game.reset()
            is_terminal = False
            reward = 0
            action = self.choose_action(state, q)
            while not is_terminal:
                next_state, reward, is_terminal = game.one_move(action)
                next_action = self.choose_action(next_state, q)
                q[state][action].average += self.alpha*(reward + self.gamma
                                    * q[next_state][next_action].average-q[state][action].average)
                state = next_state
                action = next_action
        return q

    def expected_sarsa(self, episodes):
        game = self.game
        q = game.q_initializer()
        for i in tqdm(range(episodes)):
            state = game.reset()
            is_terminal = False
            reward = 0
            action = self.choose_action(state, q)
            while not is_terminal:
                next_state, reward, is_terminal = game.one_move(action)
                next_action = self.choose_action(next_state, q)
                candidate_lst = list(q[next_state].values()) + [q[next_state][next_action].average]
                prob_dist = len(q[next_state]) * [self.epsilon]+\
                            [(len(q[snext_state]) * (1 - self.epsilon))]
                prob_list = [prob/sum(prob_dist) for prob in prob_dist]
                target = np.dot(candidate_lst, prob_list)
                q[state][action].average += self.alpha*(reward + self.gamma
                                    * target - q[state][action].average)
                state = next_state
                action = next_action
        return q

if __name__ == '__main__':
    from game.blackjack import BlackJack

    test = BlackJack()
    test_q = QLearner(test, 0.1, 0.1, 1)
    q1 = test_q.sarsa(3*10**5)
