from tabular_learning import Agent
import numpy as np
from tqdm import tqdm
from collections import defaultdict


class NStepTD(Agent):

    def __init__(self, game, n, gamma=1, alpha=10 ** -4, epsilon=0.1, q=None):
        self.game = game
        self.n = n
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.q = q

    def fit(self):

        q = self.game.q_initializer()
        for _ in tqdm(range(5 * 10 ** 6)):
            T = float('INF')
            t = 0
            tao = 0
            state = game.reset()
            action = self.epsilon_greedy(q[state])[0]
            # is_terminal = False
            state_dct = {0: state}
            action_dct = {0: action}
            reward_dct = {}

            while tao < T:
                if t < T:
                    next_state, reward, is_terminal = game.one_move(action)
                    state_dct[t+1] = next_state
                    reward_dct[t+1] = reward
                    if is_terminal:
                        T = t + 1
                    else:
                        action = self.epsilon_greedy(q[next_state])[0]
                        action_dct[t+1] = action
                tao = t - self.n + 1
                if tao >= 0:
                    G = 0
                    for i in range(tao+1, min(tao+self.n, T)+1):
                        G += self.gamma ** (i - tao - 1) * reward_dct[i]
                        if (tao + self.n) < T:
                            G = G + self.gamma ** self.n * \
                                q[state_dct[tao+self.n]][action_dct[tao+self.n]]
                        # if G > 1:
                        #     print('haha')
                        q[state_dct[tao]][action_dct[tao]] += self.alpha * (
                            G - q[state_dct[tao]][action_dct[tao]]
                        )
                t = t + 1
        self.q = q


if __name__ == '__main__':
    from game.gambler import Gambler
    game = Gambler(p_h=0.4, goal=15)
    learner = NStepTD(game, 1)
    learner.fit()
    print(learner.q)
