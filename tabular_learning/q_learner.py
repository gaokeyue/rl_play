from tabular_learning import Agent
import numpy as np
from tqdm import tqdm


class QLearner(Agent):
    """q_learning, sarsa, expected sarsa"""

    def __init__(self, game, alpha=10 ** -4, epsilon=0.1, n_trials=5 * 10 ** 4):
        super().__init__(game, n_trials)
        self.alpha = alpha
        self.epsilon = epsilon

    def epsilon_greedy(self, state, q):
        if np.random.random() < self.epsilon:
            chosen_action = np.random.choice(self.game.available_actions(state))
        else:
            chosen_action = np.random.choice(self.greedy_pick(q[state], find_one=False))
        return chosen_action

    def crude_run(self, episodes=5 * 10 ** 5, q0=None, soi=None):
        game = self.game
        q = game.q_initializer() if q0 is None else q0
        for _ in tqdm(range(episodes)):
            if soi is None:
                state = game.reset()
            else:
                state = game.reset(np.random.choice(soi))
            is_terminal = False
            while not is_terminal:
                action = self.epsilon_greedy(state, q)
                next_state, reward, is_terminal = game.one_move(action)
                q_next = max(q[next_state].values())
                q[state][action] += self.alpha * (reward + game.gamma * q_next - q[state][action])
                state = next_state
        return q

    def sarsa(self, episodes=50 * 10 ** 5):
        game = self.game
        q = game.q_initializer()
        for i in tqdm(range(episodes)):
            state = game.reset()
            is_terminal = False
            action = self.epsilon_greedy(state, q)
            while not is_terminal:
                next_state, reward, is_terminal = game.one_move(action)
                next_action = self.epsilon_greedy(next_state, q)
                q[state][action].average += self.alpha * (reward + self.gamma
                                                          * q[next_state][next_action].average - q[state][
                                                              action].average)
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
            action = self.epsilon_greedy(state, q)
            while not is_terminal:
                next_state, reward, is_terminal = game.one_move(action)
                next_action = self.epsilon_greedy(next_state, q)
                candidate_lst = list(v.average for v in q[next_state].values()) + [q[next_state][next_action].average]
                prob_dist = len(q[next_state]) * [self.epsilon] + \
                            [(len(q[next_state]) * (1 - self.epsilon))]
                prob_list = [prob / sum(prob_dist) for prob in prob_dist]
                target = np.dot(candidate_lst, prob_list)
                q[state][action].average += self.alpha * (reward + self.gamma
                                                          * target - q[state][action].average)
                state = next_state
                action = next_action
        return q


if __name__ == '__main__':
    from game.blackjack import BlackJack
    import pandas as pd
    test_q = QLearner(BlackJack())
    q1 = test_q.crude_run(5 * 10 ** 6)
    df = pd.DataFrame(q1).T
    print(df.head())
    df.to_csv('../qleaning.csv')
