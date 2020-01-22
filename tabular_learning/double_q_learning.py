# from game.Blackjack import BlackJack
import random
import math
import copy


class DoubleQLearning:
    def __init__(self, game, learn_alpha=0.1, decay_gamma=0.99, vary_epsilon=0.1):
        self.game = game
        self.learn_alpha, self.decay_gamma, self.vary_epsilon = learn_alpha, decay_gamma, vary_epsilon
        self.state_last,  self.state_current, self.state_next = None, self.game.reset(), None
        self.ends = False
        self.action = None
        self.reward = 0
        self.q1, self.q2 = {}, {}

    def q_initializer(self):
        # print(type(self.game.available_actions()), self.game.available_actions())
        self.q1[self.state_current] = dict(zip(self.game.available_actions(), [0] * len(self.game.available_actions())))
        self.q2 = copy.deepcopy(self.q1)

    def action_decide(self, p_epsilon=None, p_state=None):
        if p_epsilon is None:
            p_epsilon, p_state = self.vary_epsilon, self.state_current
        variation = random.random()
        if variation < p_epsilon:
            actions = self.game.available_actions()
        else:
            q_values = self.q1[p_state]
            q_keys = sorted(q_values, key=q_values.__getitem__)
            actions = [x for x in self.game.available_actions() if math.isclose(q_values[x], q_values[q_keys[0]])]
        return random.choice(actions)

    def q_update(self):
        if
        qm, qn = random.choice([self.q1, self.q2], size=2, replace=True, p=None)
        action_in_q = self.action_decide(0, self.state_current)
        qm[self.state_last][action_in_q] *= (1 - self.learn_alpha)
        qm[self.state_last][action_in_q] += self.learn_alpha * (self.reward + self.decay_gamma * qn[self.state_current][self.action_in_q])

    def double_q_learning(self, episodes, times):
        for p_time in range(times):
            self.q_initializer()
            for p_episode in range(episodes):
                while self.ends is False:
                    self.state_last = self.state_current
                    self.state_current = self.game.reset()
                    self.action = self.action_decide()
                    self.state_current, self.reward, self.ends = self.game.one_move(self.action)
                    self.q_update()


if __name__ == '__main__':
    # from game.gambler import Gambler
    # test_item = Gambler()
    from game.Blackjack import BlackJack
    test_item = BlackJack()
    test_tool = DoubleQLearning(test_item)
    test_tool.double_q_learning(3000, 1)
    print(test_tool.q1, test_tool.q2)
