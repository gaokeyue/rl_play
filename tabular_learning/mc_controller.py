from tabular_learning import Agent
from random import random, choice
from utils import Averager, compare, convex_comb
from tqdm import tqdm
#
from game.blackjack import BlackJack
from game.gambler import Gambler
from game.jacks_car_rental import JacksCarRental
import pandas as pd
import numpy as np
import os
import pickle


class MC(Agent):

    def __init__(self, game, epsilon=0.1):
        """
        :param game: a game (Game)
        :param epsilon: a parameter used in greedy_pick method (int/float, 0 <= epsilon < 1)
        """
        self.game = game
        self.epsilon = epsilon
        self.prob_b = {state: {action: 0 for action in game.available_actions(state)}
                       for state in game.state_space}

    @staticmethod
    def greedy_selection(value_dict, find_one=True, thresh=0):
        """
        :param value_dict: an action value function with the state fixed (dict) or a state value function (dict)
        :param find_one: a switch (bool, True/False to return one/all of the best actions)
        :param thresh: a threshold value used in float comparison (float)
        :return:
        """
        if find_one:  # find one maximizer
            return max(value_dict, key=value_dict.get)
        else:
            best_score = -float('inf')
            best_results = []
            for key, value in value_dict.items():
                score = value.average if isinstance(value, Averager) else value
                flag = compare(score, best_score, thresh)
                if flag is 1:  # a strictly better action is found
                    best_score = score
                    best_results = [key]
                elif flag is 0:  # an action which ties the best action is found
                    best_results.append(key)
            return best_results

    def q_initializer(self, use_averager=False, default_value=0):
        """
        :param use_averager: a switch (bool, False to use int/float, True to use a Averager class)
        :param default_value: the initial state-action value (int or float)
        :return: an initial state-action value function (dict of dict)
        """
        game = self.game
        if use_averager:
            q = {state: {action: Averager(default_value) for action in game.available_actions(state)}
                 for state in game.state_space}
        else:
            q = {state: {action: default_value for action in game.available_actions(state)}
                 for state in game.state_space}
        return q

    def v_initializer(self, use_averager=False, default_value=0):
        """
        :param use_averager: a switch (bool, False to use int/float, True to use a Averager class)
        :param default_value: the initial state-action value (int or float)
        :return: an initial state value function (dict of dict)
        """
        game = self.game
        if use_averager:
            v = {state: Averager(default_value) for state in game.state_space}
        else:
            v = {state: default_value for state in game.state_space}
        return v

    def choose_action_on_q(self, state, q_dict, off_policy=False):
        """
        :param state: a state
        :param q_dict: action value function (dict of dict)
        :param off_policy: a switch (bool, False/True for off_policy/on_policy)
        :return: an action chosen by epsilon_greedy method
        """
        game = self.game
        explored = False
        if random() < self.epsilon:
            action_chosen = choice(game.available_actions(state))
            explored = True
        else:
            action_chosen = choice(self.greedy_selection(q_dict[state], find_one=False))
        if off_policy:
            best_action = choice(self.greedy_selection(q_dict[state], find_one=False)) if explored else action_chosen
            for action in game.available_actions(state):
                if action == best_action:
                    self.prob_b[state][action] = 1 - self.epsilon + self.epsilon / len(game.available_actions(state))
                else:
                    self.prob_b[state][action] = self.epsilon / len(game.available_actions(state))
        return action_chosen

    def choose_action_on_v(self, state, v_dict):
        """
        :param state: a state
        :param v_dict: state value function (dict)
        :return: an action chosen by epsilon_greedy method
        """
        game = self.game
        if random() < self.epsilon:
            action_chosen = choice(game.available_actions(state))
        else:
            after_states = tuple(game.half_move(action, state) for action in game.available_actions(state))
            v_dict_selected = {after_state: v_dict[after_state] for after_state in after_states}
            after_state = choice(self.greedy_selection(v_dict_selected, find_one=False))
            action_chosen = game.half_move_reverse(after_state, state)
        return action_chosen

    def episode_generator(self, value_dict, value_type='q', state0=None, policy=None, off_policy=False, length=None):
        """
        :param value_dict: a state-action value function (dict of dict) or a state value function (dict)
        :param state0: a state used to generate the episode
        :param policy: a policy used to determine the action from a certain state (dict)
        :param off_policy: a switch (bool, False for off_policy, True for on_policy)
        :param value_type: 'q' for state-action value function, 'v' for state value function
        :return: an episode including states, actions and rewards (list, list, list)
        """
        game = self.game
        state = game.reset(state0) if state0 else game.reset()
        state_ls, action_ls, reward_ls = [], [], []
        is_terminal = False
        i = 0
        while not is_terminal:
            if length is not None:
                i += 1
                if i == length:
                    break
            state_ls.append(state)
            if value_type == 'q':
                action = policy[state] if policy else self.choose_action_on_q(state, value_dict, off_policy)
            elif value_type == 'v':
                action = policy[state] if policy else self.choose_action_on_v(state, value_dict)
            else:
                print('Something wrong with value_type!')
                exit()
            action_ls.append(action)
            state, reward, is_terminal = game.one_move(action)
            reward_ls.append(reward)
        return state_ls, action_ls, reward_ls

    def on_policy_mc_exploring_start(self, n_episodes=5*10**5, policy0=None, q0=None, epi_length=None):
        """
        :param n_episodes: the number of episodes (int)
        :param policy0: a policy used in the first episode (None or dict)
        :param q0: an initial state-action value function (None or dict of dict)
        :return: a state-action value function (dict of dict)
        """
        self.epsilon = 0
        game = self.game
        if q0 is None:
            q_dict = self.q_initializer(use_averager=True, default_value=1)
            sa_oi = None
        else:
            q_dict = {}
            sa_oi = []
            for state, action_value_dict in q0.items():
                q_dict[state] = {}
                for action, value in action_value_dict.items():
                    q_dict[state][action] = Averager(value)
                    if len(action_value_dict) > 1:
                        sa_oi.append((state, action))
        for episode in tqdm(range(n_episodes)):
            if q0 is None:
                state0 = game.reset()
                action0 = choice(game.available_actions(state0))
            else:
                state0, action0 = choice(sa_oi)
                game.reset(state0)
            state1, reward0, is_terminal = game.one_move(action0)
            state_ls, action_ls, reward_ls = [state0], [action0], [reward0]
            if not is_terminal:
                policy = None if episode else policy0
                state_ls_2, action_ls_2, reward_ls_2 = self.episode_generator(q_dict, state0=state1,
                                                                              policy=policy, length=epi_length)
                state_ls.extend(state_ls_2)
                action_ls.extend(action_ls_2)
                reward_ls.extend(reward_ls_2)
            g = 0
            for s, a, r in zip(reversed(state_ls), reversed(action_ls), reversed(reward_ls)):
                g = game.gamma * g + r
                if s in q_dict:
                    if a in q_dict[s]:  # if next state not encountered before, no need for updating
                        q_dict[s][a].add_new(g)  # using every-visit
        q_dict = {state: {action: q.average for action, q in value.items()} for state, value in q_dict.items()}
        return q_dict

    def on_policy_mc_epsilon_soft(self, n_episodes=5*10**5, q0=None):
        """
        :param n_episodes: the number of episodes (int)
        :param q0: an initial state-action value function (dict of dict)
        :return: a state-action value function (dict of dict)
        """
        game = self.game
        if q0 is None:
            q_dict = self.q_initializer(use_averager=True, default_value=1)
            s_oi = None
        else:
            q_dict = {}
            s_oi = []
            for state, action_value_dict in q0.items():
                q_dict[state] = {}
                for action, value in action_value_dict.items():
                    q_dict[state][action] = Averager(value)
                if len(action_value_dict) > 1:
                    s_oi.append(state)
        for _ in tqdm(range(n_episodes)):
            if q0 is None:
                state0 = game.reset()
            else:
                state0 = choice(s_oi)
                game.reset(state0)
            state_ls, action_ls, reward_ls = self.episode_generator(q_dict, state0=state0)
            g = 0
            for s, a, r in zip(reversed(state_ls), reversed(action_ls), reversed(reward_ls)):
                g = game.gamma * g + r
                if s in q_dict:
                    if a in q_dict[s]:  # if next state not encountered before, no need for updating
                        q_dict[s][a].add_new(g)  # using every-visit
        q_dict = {state: {action: q.average for action, q in value.items()} for state, value in q_dict.items()}
        return q_dict

    def off_policy_mc(self, n_episodes=5*10*5, q0=None):
        """
        :param n_episodes: the number of episodes (int)
        :param q0: an initial state-action value function (dict of dict)
        :return: a state-action value function (dict of dict)
        """
        game = self.game
        if q0 is None:
            q_dict = self.q_initializer(default_value=0)
            s_oi = None
        else:
            q_dict = q0
            s_oi = []
            for state, action_value_dict in q0.items():
                if len(action_value_dict) > 1:
                    s_oi.append(state)
        c_fun = {state: {action: 0 for action in game.available_actions(state)}
                 for state in game.state_space}
        for _ in tqdm(range(n_episodes)):
            if q0 is None:
                state0 = game.reset()
            else:
                state0 = choice(s_oi)
                game.reset(state0)
            state_ls, action_ls, reward_ls = self.episode_generator(q_dict, state0=state0, off_policy=True)
            g = 0
            w = 1
            for s, a, r in zip(reversed(state_ls), reversed(action_ls), reversed(reward_ls)):
                g = game.gamma * g + r
                c_fun[s][a] += w
                if s in q_dict:
                    if a in q_dict[s]:  # if next state not encountered before, no need for updating
                        q_dict[s][a] = convex_comb(q_dict[s][a], g, w / c_fun[s][a])
                    best_action = choice(self.greedy_selection(q_dict[s], find_one=False))
                    if best_action != a:
                        break
                w *= 1 / self.prob_b[s][a]
        return q_dict


if __name__ == '__main__':

    project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_dir = os.path.join(project_dir, 'data')
    """test0 - BlackJack"""
    # game0 = BlackJack()
    # agent0 = MC(game0)
    # n_episodes = 1 * 10 ** 7
    # # q1 = agent0.on_policy_mc_exploring_start(n_episodes=n_episodes, policy0=game0.policy_initializer(), q0=None)
    # # q1 = agent0.on_policy_mc_epsilon_soft(n_episodes=n_episodes, q0=None)
    # q1 = agent0.off_policy_mc(n_episodes=n_episodes, q0=None)
    # q_df = pd.DataFrame.from_dict(q1, orient='index')
    # q_df.sort_index(level=[0, 1, 2], inplace=True)
    # q_df.to_csv(data_dir + '/q3.csv')
    """test1 - Gambler"""
    # game1 = Gambler(goal=15)
    # agent1 = MC(game1, epsilon=0.1)
    # n_episodes = 5 * 10 ** 6
    # # q1 = agent1.on_policy_mc_exploring_start(n_episodes=n_episodes, policy0=None, q0=None)
    # # q1 = agent1.on_policy_mc_epsilon_soft(n_episodes=n_episodes, q0=None)
    # q1 = agent1.off_policy_mc(n_episodes=n_episodes, q0=None)
    # q_df = pd.DataFrame.from_dict(q1, orient='index')
    # q_df.sort_index(level=0, inplace=True)
    # q_df.to_csv(data_dir + '/qq3_-1.csv')
    """test2 - JacksCarRental"""
    game2 = JacksCarRental()
    agent2 = MC(game2, epsilon=0)
    # n_episodes = 5 * 10 ** 5
    # q1 = agent2.on_policy_mc_exploring_start(n_episodes=n_episodes, epi_length=100)
    # v1 = {state: max(act_dict.values()) for state, act_dict in q1.items()}
    # policy1 = {state: max(act_dict, key=act_dict.get) for state, act_dict in q1.items()}
    # v_np = np.zeros((21, 21))
    # policy_np = np.zeros((21, 21))
    # for i in range(21):
    #     for j in range(21):
    #         v_np[i, j] = v1[(i, j)]
    #         policy_np[i, j] = policy1[(i, j)]
    # with open(data_dir + '/jacks_v.pkl', 'wb') as f:
    #     pickle.dump(v1, f)
    with open(data_dir + '/jacks_v.pkl', 'rb') as f:
        v1 = pickle.load(f)
    s_ls, a_ls, r_ls = agent2.episode_generator(v1, value_type='v', length=100)
    #
    # print('haha')
    #
