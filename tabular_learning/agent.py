from collections import defaultdict
import utils
from game.blackjack import BlackJack

import numpy as np


class Agent:
    def __init__(self, game):
        self.game = game

    @staticmethod
    def greedy_pick(action_value_dict, find_one=False, thresh=0):
        if find_one:  # find one argmaxizer
            return max(action_value_dict, key=action_value_dict.get)
        else:
            best_score = -float('inf')
            best_actions = []
            for action, value in action_value_dict.items():
                flag = utils.compare(value, best_score, thresh)
                if flag is 1:  # a strictly better action is found
                    best_score = value
                    best_actions = [action]
                elif flag is 0:  # an action which ties the best action is found
                    best_actions.append(action)
            return best_actions

    def policy_run(self, policy, *reset_args):
        state = self.game.reset(*reset_args)
        state_ls = []
        action_ls = []
        reward_ls = []
        is_terminal = False
        while not is_terminal:
            state_ls.append(state)
            self.game._action = policy[state]
            action_ls.append(self.game.action)
            state, reward, is_terminal = self.game.one_move(self.game.action)
            reward_ls.append(reward)
        return state_ls, action_ls, reward_ls

    def policy_eval_on(self, policy, n_episodes=10 ** 5):
        """Using Monte-Carlo to evaluate a given policy.
        :param policy (dict) -- only consider deterministic policy, action = policy[state]
        :param n_episodes (int)
        """
        value_fun = {state: utils.Averager() for state in policy}
        for _ in range(n_episodes):
            state_ls, _, reward_ls = self.policy_run(policy)
            v = 0
            for s, r in zip(reversed(state_ls), reversed(reward_ls)):
                v = self.game.gamma * v + r
                value_fun[s].add_new(v)
        return value_fun

    def policy_eval_off(self, policy):
        pass


if __name__ == '__main__':

    #
    game1 = BlackJack()
    policy1 = {key: 'stand' if key[1] >= 20 else 'hit' for key in game1.q_initializer().keys()}
    agent1 = Agent(game1)
    v1 = agent1.policy_eval_on(policy1, n_episodes=5*10**5)
    v_usable_ace = np.zeros((10, 10))
    v_no_usable_ace = np.zeros((10, 10))
    for key, value in v1.items():
        if key[2]:
            i = key[1] - 12
            j = 0 if key[0] is 'A' else (game1._cards_count[key[0]] - 1)
            v_usable_ace[i, j] = value.average  # try to compare with RL book fig 5.1(b)
        else:
            i = key[1] - 12
            j = 0 if key[0] is 'A' else (game1._cards_count[key[0]] - 1)
            v_no_usable_ace[i, j] = value.average  # try to compare with RL book fig 5.1(d)
    print('haha')
    #
