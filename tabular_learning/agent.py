from collections import defaultdict, Counter
import random
from game.blackjack import BlackJack
import utils
import operator
import pandas as pd


class Agent:
    def __init__(self, game):
        self.game = game

    @staticmethod
    def greedy_pick(action_value_dict, find_one=True, thresh=0):
        if find_one:  # find one maximizer
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

    def policy_run(self, policy, state0=None):
        state = self.game.reset(state0)
        state_ls = []
        reward_ls = []
        is_terminal = False
        while not is_terminal:
            state_ls.append(state)
            action = policy[state]
            state, reward, is_terminal = self.game.one_move(action)
            reward_ls.append(reward)
        return state_ls, reward_ls

    def policy_eval_on(self, policy, n_episodes=5):
        """Using Monte-Carlo to evaluate a given policy.
        :param policy (dict) -- only consider deterministic policy, action = policy[state]
        :param n_episodes -- unit is Wan
        """
        n_episodes *= 10 ** 4
        value_fun = {state: utils.Averager() for state in policy}
        soi = list(policy)
        for i in range(n_episodes):
            state0 = random.choice(soi)
            state_ls, reward_ls = self.policy_run(policy, state0)
            v = 0
            for s, r in zip(reversed(state_ls), reversed(reward_ls)):
                v = self.game.gamma * v + r
                value_fun[s].add_new(v)
        return value_fun

    def q_eval(self, q):
        # first get state value function
        state_value = {}
        pi_star = {}
        for state, action_value in q.items():
            a, v = max(action_value.items(), key=operator.itemgetter(1))
            state_value[state] = v
            pi_star[state] = a
        self.policy_eval_on(pi_star, )

    def policy_eval_off(self, policy):
        pass


def brutal_test(n=5 * 10 ** 5):
    player = Agent(BlackJack())
    state0 = ("K", 16, False)
    stick_cnt = Counter()
    print(f"Initial state is {state0}")
    print("-" * 30)
    print("If we choose stick")
    for _ in range(n):
        player.game.reset(state0)
        state, reward, is_terminal = player.game.one_move('stick')
        assert is_terminal
        stick_cnt[reward] += 1
    print(stick_cnt)
    print(round(sum(r * freq / n for r, freq in stick_cnt.items()), 6))
    print('-' * 30)
    print("If we choose hit then stick")
    hit_cnt = Counter()
    for _ in range(n):
        player.game.reset(state0)
        state, reward, is_terminal = player.game.one_move('hit')
        if is_terminal:
            hit_cnt[reward] += 1
        else:
            state, reward, is_terminal = player.game.one_move('stick')
            assert is_terminal
            hit_cnt[reward] += 1
    print(hit_cnt)
    print(round(sum(r * freq / n for r, freq in hit_cnt.items()), 6))


def get_q_star(n_episodes=10 ** 5):
    player = Agent(BlackJack())
    # states of interest
    soi = [("K", i, False) for i in range(12, 22)]
    pi_star = {state: 'hit' if state[1] <= 16 else 'stand' for state in soi}
    q_star = defaultdict(dict)
    game = player.game
    for state in soi:
        for action in ['hit', 'stand']:
            action_value = utils.Averager()
            for _ in range(n_episodes):
                game.reset(state)
                tmp_a = action
                while True:
                    tmp_s, reward, is_terminal = game.one_move(tmp_a)
                    if is_terminal:
                        action_value.add_new(reward)
                        break
                    else:
                        tmp_a = pi_star[tmp_s]
            print(f"({state[1]}, {action}) is {action_value.average:.6f}")
            q_star[state[1]][action] = action_value.average
    return q_star


if __name__ == '__main__':
    print('haha')
    n = 1 * 10 ** 5
    q_star = get_q_star(n)
    df = pd.DataFrame(q_star).T
    print(df)
    brutal_test(n)
