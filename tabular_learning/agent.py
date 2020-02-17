import utils
import os
from collections import defaultdict, Counter
import random
from operator import itemgetter
import numpy as np
import pandas as pd
from tqdm import tqdm


class Agent:
    def __init__(self, game):
        self.game = game
        self.project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        self.data_dir = os.path.join(project_dir, 'data')

    def q_initializer(self):
        if self.game.state_space is None:
            return defaultdict(lambda: defaultdict(int))
        game = self.game
        q = {}
        for state in game.state_space:
            actions = game.available_actions(state)
            action_value = {action: 0 for action in actions}
            if len(action_value) == 0:  # actions is empty, i.e. state is terminal
                action_value[None] = 0
            q[state] = action_value
        return q

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

    @staticmethod
    def epsilon_greedy(action_value_dict, epsilon=.1):
        """return action and its current value"""
        if random.random() < epsilon:
            return random.choice(list(action_value_dict.items()))
        else:
            return max(list(action_value_dict.items()), key=itemgetter(1))

    @staticmethod
    def q2pi(q):
        """If state action value Q(S, A) is given, induce the corresponding greedy policy."""
        return {state: Agent.greedy_pick(action_value_dict)
                for state, action_value_dict in q.items()}

    @staticmethod
    def compare_dict(d1, d2):
        """Find keys in d1 such that the corresponding values differ in d1 and d2"""
        return [k for k in d1 if d1[k] != d2[k]]

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

    def get_q_pi_sa(self, policy, state, action, n_episodes=10 ** 5):
        """Given a policy pi, evaluate Q_pi(state, action). Note that action not
        necessarily equals pi[state]
        """
        game = self.game
        result = utils.Averager()
        for _ in tqdm(range(n_episodes)):
            game.reset(state)
            tmp_a = action
            total_return = 0
            t = 0
            while True:
                tmp_s, reward, is_terminal = game.one_move(tmp_a)
                total_return += reward * game.gamma ** t
                t += 1
                if is_terminal:
                    result.add_new(total_return)
                    break
                else:
                    tmp_a = policy[tmp_s]
        return result.average

    def policy_eval(self, policy, state_action_oi=None, n_episodes=10**5):
        """Using Monte-Carlo to evaluate Q_pi(S, A) for (S, A) of interest.
        :param policy (dict) -- only consider deterministic policy, action = policy[state]
        :param state_action_oi (dict) -- (state, action) pair of interest,
        if None, then evaluate every possible state-action pair where state in policy
        """
        if state_action_oi is None:
            state_action_oi = {state: self.game.available_actions(state) for state in policy}
        q = {}
        for state, actions in state_action_oi.items():
            action_value_dict = {}
            for action in actions:
                v = self.get_q_pi_sa(policy, state, action, n_episodes)
                action_value_dict[action] = v
                # print(F"{state}, {action} --> {v}")
            q[state] = action_value_dict
        return q

    def fine_tune(self, q_star, max_iter=10):
        sa_oi = {}
        pi_star = Agent.q2pi(q_star)
        for state, action_value_dict in q_star.items():
            if len(action_value_dict) > 1:
                sa_oi[state] = list(action_value_dict)
        for i in range(max_iter):
            print(f"The {i + 1}th trial.")
            q_hat = self.policy_eval(pi_star, sa_oi, n_episodes)
            q_hat = {}
            for state, actions in sa_oi.items():

                q_star[state]
                n_episodes = 10 ** 5
                for action in actions:
                    v = self.get_q_pi_sa(pi_star, state, action, n_episodes)
                    q_hat[state]

            pi_hat = self.q2pi(q_hat)
            problem_states = self.compare_dict(pi_hat, pi_star)
            if problem_states:
                self.n_trials *= 2
                print(f"problem states are {problem_states}")
                sa_oi = {state: list(q_hat[state]) for state in problem_states}
                pi_star.update(pi_hat)
            else:
                print("Done!!!!")
                return pi_star
        else:
            print(f"Exceeds maximum iteration = {max_iter}")

    def crude_run(self):
        q_star = self.q_learning(n_episodes=10 ** 7, alpha=10 ** -3)
        for prune_thresh in [.2, .1, .06]:
            q_star = action_prune(q_star, prune_thresh)
            # check the pruning is correct
            q_star = self.q_learning(q0=q_star, n_episodes=10 ** 7, alpha=10 ** -4)
            # show_q(q_star)
        with open(os.path.join(self.data_dir, 'q_star.pkl'), 'wb') as file:
            pickle.dump(q_star, file)
        df = pd.DataFrame(q_star).T
        df = df.reindex(columns=['hit', 'stand'])
        df.sort_index(inplace=True)
        df.to_csv('~/Desktop/q_star.csv')
        return q_star

    def aug_q(self, alpha=10 ** -3, eps=0.1):
        q = {}
        successor = {}  # next state
        im_r = {}  # immediate reward
        predecessor = {}  # previous state
        # optimistic start
        game = self.game
        state0 = game.reset()
        is_terminal = False
        while not is_terminal:
            avail_actions = game.available_actions()
            if state0 not in q:  # a new state is encountered
                q[state0] = {action: 1 for action in avail_actions}
                successor[state0] = {action: Counter() for action in avail_actions}
                im_r[state0] = {action: Counter() for action in avail_actions}
            if random.random() <= eps:
                action0 = random.choice(avail_actions)
            else:
                action0 = max(q[state0], key=q[state0].get)
            state1, reward, is_terminal = game.one_move(action0)
            successor[state0][action0][state1] += 1
            im_r[state0][action0][reward] += 1
            if state1 not in predecessor:
                predecessor[state1] = Counter()
            predecessor[state1][state0] += 1
            q[state0][action0] += alpha * (
                    reward + game.gamma * max(q[state1].values()) - q[state0][action0])


def show_q(q):
    # df = pd.DataFrame(q).T
    # df.sort_index(inplace=True)
    #
    states = sorted(q.keys())
    for state in states:
        action_dict = q[state]
        if len(action_dict) > 1:
            for action, v in action_dict.items():
                print(f"({state}, {action}) = {q[state][action]:.6f}")


def action_prune(q_dict, thresh):
    n0 = 0
    x = 0
    for state, action_value_dict in q_dict.items():
        n0 += len(action_value_dict)
        best = max(action_value_dict.values())
        q_dict[state] = {a: v for a, v in action_value_dict.items()
                         if best - v < thresh}
        x += len(q_dict[state])
    print(f"{n0} state-action pairs before pruning")
    print(f"{n0 - x} state-action pairs pruned")
    return q_dict


if __name__ == '__main__':
    import os

    project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_dir = os.path.join(project_dir, 'data')
    from game.blackjack import BlackJack
    import pickle

    player = Agent(BlackJack())
    # q_star = player.crude_run()
    pi_star = player.fine_tune()
