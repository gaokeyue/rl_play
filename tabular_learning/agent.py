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

    def q_initializer(self):
        """may to be deprecated in the future since the default action value is set to 1. """
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
    def greedy_action(action_value_dict, find_one=True, thresh=0):
        """Just return the greedy action(s)."""
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
        """return a tuple of action and its current value"""
        if random.random() < epsilon:
            return random.choice(list(action_value_dict.items()))
        else:
            return max(list(action_value_dict.items()), key=itemgetter(1))

    @staticmethod
    def q2pi(q):
        """If state action value Q(S, A) is given, induce the corresponding greedy policy."""
        return {state: Agent.greedy_action(action_value_dict)
                for state, action_value_dict in q.items()}

    @staticmethod
    def compare_dict(d1, d2):
        """The keys in d1 form a subset of those in d2.
        Find keys in d1 such that the corresponding values differ in d1 and d2"""
        return [k for k in d1 if d1[k] != d2[k]]

    @staticmethod
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

    def policy_run(self, policy, state0=None):
        """Run the game according to a given policy"""
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
        """Given a policy pi, evaluate Q_pi(state, action). Note that action is not
        necessarily policy[state]
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

    def policy_eval(self, policy, state_action_oi=None, n_episodes=10 ** 5):
        """Using Monte-Carlo to evaluate Q_pi(S, A) for (S, A) of interest.
        :param policy (dict) -- only consider deterministic policy, action = policy[state]
        :param state_action_oi (dict) -- (state, action) pair of interest,
        if None, then evaluate every possible state-action pair where state in policy
        """
        if state_action_oi is None:
            state_action_oi = {state: self.game.available_actions(state) for state in policy}
        q_dct = {}
        for state, actions in state_action_oi.items():
            action_value_dict = {}
            for action in actions:
                v = self.get_q_pi_sa(policy, state, action, n_episodes)
                action_value_dict[action] = v
                # print(F"{state}, {action} --> {v}")
            q_dct[state] = action_value_dict
        return q_dct

    def fine_tune(self, q_star, max_iter=10):
        """Given state-action value dictionary q_star(whose actions are usually already pruned to make this routine
        computationally feasible), evaluate q_pi(S, A) for each state under which best actions are to be determined,
        and pick the best action(s).
        """
        sa_oi = {}  # state-action pair of interest
        pi_star = Agent.q2pi(q_star)
        for state, action_value_dict in q_star.items():
            if len(action_value_dict) > 1:
                sa_oi[state] = list(action_value_dict)
        for i in range(max_iter):
            print(f"The {i + 1}th trial.")
            q_hat = {}
            for state, actions in sa_oi.items():
                # for each state, compute the difference between difference action values.
                v_ls = q_star[state].values()
                d = max(v_ls) - min(v_ls)
                # 3/sqrt(n_episodes) = d
                n_episodes = max(5*10**4, int((3 / d) ** 2))  # at least 5W
                q_hat[state] = {action: self.get_q_pi_sa(pi_star, state, action, n_episodes) for action in actions}
            pi_hat = self.q2pi(q_hat)
            problem_states = self.compare_dict(pi_hat, pi_star)
            if problem_states:
                print(f"problem states are {problem_states}")
                sa_oi = {state: actions for state, actions in sa_oi.items() if state in problem_states}
                pi_star.update(pi_hat)
                q_star.update(q_hat)
            else:
                print("Done!!!!")
                return pi_star
        else:
            print(f"Exceeds maximum iteration = {max_iter}")

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


if __name__ == '__main__':
    import os

    project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_dir = os.path.join(project_dir, 'data')

