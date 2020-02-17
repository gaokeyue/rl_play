from tabular_learning import Agent
from random import random, choice
from utils import Averager, compare
from tqdm import tqdm
#
from game.blackjack import BlackJack
import pandas as pd
import os


class MC(Agent):

    def __init__(self, game, epsilon=0.1):
        """
        :param game: a game (Game)
        :param epsilon: a parameter used in greedy_pick method (int/float, 0 <= epsilon < 1)
        """
        self.game = game
        self.epsilon = epsilon
        self.prob_b = {state: {action: 0 for action in game.available_actions(state)}
                       for state in self.game.state_space}
        self.act_freq = {state: {action: 0 for action in game.available_actions(state)}
                         for state in self.game.state_space}

    @staticmethod
    def greedy_action(action_value_dict, find_one=True, thresh=0):
        """
        :param action_value_dict: the action value function with the state fixed (dict)
        :param find_one: a switch (bool, True/False to return one/all of the best actions)
        :param thresh: a threshold value used in float comparison (float)
        :return:
        """
        if find_one:  # find one maximizer
            return max(action_value_dict, key=action_value_dict.get)
        else:
            best_score = -float('inf')
            best_actions = []
            for action, value in action_value_dict.items():
                score = value.average if isinstance(value, Averager) else value
                flag = compare(score, best_score, thresh)
                if flag is 1:  # a strictly better action is found
                    best_score = score
                    best_actions = [action]
                elif flag is 0:  # an action which ties the best action is found
                    best_actions.append(action)
            return best_actions

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

    def choose_action(self, state, q_dict, off_policy=False):
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
            action_chosen = choice(self.greedy_action(q_dict[state], find_one=False))
        if off_policy:
            best_action = choice(self.greedy_action(q_dict[state], find_one=False)) if explored else action_chosen
            for action in game.available_actions(state):
                if action == best_action:
                    self.prob_b[state][action] = 1 - self.epsilon + self.epsilon / len(game.available_actions(state))
                else:
                    self.prob_b[state][action] = self.epsilon / len(game.available_actions(state))
        return action_chosen

    def episode_generator(self, q_dict, state0=None, policy=None, off_policy=False):
        """
        :param q_dict: an action value function (dict of dict)
        :param state0: the state used to generate the episode
        :param policy: the policy used to determine the action from a certain state (dict)
        :param off_policy: a switch (bool, False for off_policy, True for on_policy)
        :return: the episode including states, actions and rewards (list, list, list)
        """
        game = self.game
        state = game.reset(state0)
        state_ls = []
        action_ls = []
        reward_ls = []
        is_terminal = False
        while not is_terminal:
            state_ls.append(state)
            action = policy[state] if policy else self.choose_action(state, q_dict, off_policy)
            action_ls.append(action)
            state, reward, is_terminal = game.one_move(action)
            reward_ls.append(reward)
        return state_ls, action_ls, reward_ls

    def on_policy_mc_exploring_start(self, n_episodes=5*10**5, policy0=None, states0=None, q0=None):
        """
        :param n_episodes: the number of episodes (int)
        :param policy0: the policy used in the first episode (None or dict)
        :param states0: the states used to generate episodes (None or iter)
        :param q0: the initial state-action value function (dict of dict)
        :return: a state-action value function (dict of dict)
        """
        self.epsilon = 0
        game = self.game
        q_dict = self.q_initializer(use_averager=True)
        for episode in tqdm(range(n_episodes)):
            state0 = game.reset(*states0) if states0 else game.reset()
            action0 = choice(game.available_actions(state0))
            state1, reward0, is_terminal = game.one_move(action0)
            state_ls, action_ls, reward_ls = [state0], [action0], [reward0]
            if not is_terminal:
                policy = None if episode else policy0
                state_ls_2, action_ls_2, reward_ls_2 = self.episode_generator(q_dict, state1, policy)
                state_ls.extend(state_ls_2)
                action_ls.extend(action_ls_2)
                reward_ls.extend(reward_ls_2)
            g = 0
            for s, a, r in zip(reversed(state_ls), reversed(action_ls), reversed(reward_ls)):
                g = game.gamma * g + r
                q_dict[s][a].add_new(g)  # using every-visit
        q_dict = {state: {action: q.average for action, q in value.items()} for state, value in q_dict.items()}
        return q_dict

    def on_policy_mc_epsilon_soft(self, n_episodes=5*10**5, states0=None):
        """
        :param n_episodes: the number of episodes (int)
        :param states0: the states used to generate episodes (None or iter)
        :return: a state-action value function (dict of dict)
        """
        game = self.game
        q_dict = self.q_initializer(use_averager=True)
        for _ in tqdm(range(n_episodes)):
            state0 = game.reset(*states0) if states0 else game.reset()
            state_ls, action_ls, reward_ls = self.episode_generator(q_dict, state0)
            g = 0
            for s, a, r in zip(reversed(state_ls), reversed(action_ls), reversed(reward_ls)):
                g = game.gamma * g + r
                q_dict[s][a].add_new(g)  # using every-visit
        q_dict = {state: {action: q.average for action, q in value.items()} for state, value in q_dict.items()}
        return q_dict

    def off_policy_mc(self, n_episodes=5*10*5, states0=None):
        """
        :param n_episodes: the number of episodes (int)
        :param states0: the states used to generate episodes (None or iter)
        :return: a state-action value function (dict of dict)
        """
        game = self.game
        q_dict = self.q_initializer()
        c_fun = {state: {action: 0 for action in game.available_actions(state)}
                 for state in game.state_space}
        for _ in tqdm(range(n_episodes)):
            state0 = game.reset(*states0) if states0 else game.reset()
            state_ls, action_ls, reward_ls = self.episode_generator(q_dict, state0, off_policy=True)
            g = 0
            w = 1
            for s, a, r in zip(reversed(state_ls), reversed(action_ls), reversed(reward_ls)):
                g = game.gamma * g + r
                c_fun[s][a] += w
                q_dict[s][a] += w / c_fun[s][a] * (g - q_dict[s][a])
                best_action = choice(self.greedy_action(q_dict[s], find_one=False))
                if best_action != a:
                    break
                w *= 1 / self.prob_b[s][a]
        return q_dict

    def calc_act_freq(self, q_dict):
        for state, value in q_dict.items():
            best_action = self.greedy_action(value)
            self.act_freq[state][best_action] += 1


def q_avg_test(game, controller='on_epsilon_greedy', n_episodes=5*10**5, n_test=10, state0_ls=None, q_avg=None):
    agent = MC(game)
    q_avg = q_avg if q_avg else {state: {action: Averager() for action in game.available_actions(state)}
                                 for state in game.state_space}
    for _ in range(n_test):
        if controller == 'on_exploring_start':
            q1 = agent.on_policy_mc_exploring_start(n_episodes=n_episodes, policy0=game.policy_initializer(),
                                                    states0=state0_ls)
        elif controller == 'on_epsilon_greedy':
            q1 = agent.on_policy_mc_epsilon_soft(n_episodes=n_episodes, states0=state0_ls)
        elif controller == 'off':
            q1 = agent.off_policy_mc(n_episodes=n_episodes, states0=state0_ls)
        else:
            print('there is no such mc controller!')
            exit()
        agent.calc_act_freq(q1)
        for state, value in q1.items():
            for act, q in value.items():
                q_avg[state][act].add_new(q)
    return q_avg, agent.act_freq


if __name__ == '__main__':

    project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_dir = os.path.join(project_dir, 'data')
    game = BlackJack()
    state0_ls = [(shown_card, hands_sum, usable_ace)
                 for shown_card in game.cards
                 for hands_sum in range(11, 22)
                 for usable_ace in (True, False)]
    """test0"""
    n_episodes = 1 * 10 ** 7
    agent1 = MC(game)
    q1 = agent1.on_policy_mc_exploring_start(n_episodes=n_episodes,
                                             policy0=game.policy_initializer(),
                                             states0=state0_ls)
    q_df = pd.DataFrame.from_dict(q1, orient='index')
    q_df.sort_index(level=[0, 1, 2], inplace=True)
    q_df.to_csv(data_dir + '/q1.csv')
    """test1"""
    # n_test = 5
    # n_episodes = 1 * 10 ** 7
    # q_avg = {state: {action: Averager() for action in game.available_actions(state)}
    #          for state in game.state_space}
    # q_avg, act_freq = q_avg_test(game=game, controller='on_epsilon_greedy', n_episodes=n_episodes,
    #                              n_test=n_test, state0_ls=state0_ls, q_avg=q_avg)
    # q_df = pd.DataFrame.from_dict(q_avg, orient='index')
    # q_df.sort_index(level=[0, 1, 2], inplace=True)
    # q_df.to_csv(data_dir + '/q2_avg.csv')
    #
    # print('haha')
    #
