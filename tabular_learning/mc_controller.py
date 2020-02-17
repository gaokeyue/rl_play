from tabular_learning import Agent
from numpy.random import choice, binomial
#
from game.blackjack import BlackJack
from numpy import zeros
import utils
from tqdm import tqdm


class MC(Agent):

    def __init__(self, game, gamma=1, epsilon=0.1):
        self.game = game
        self.gamma = gamma
        self.epsilon = epsilon
        self.prob_b = {state: {action: 0 for action in game.available_actions(state)}
                       for state in self.game.state_space}

    @staticmethod
    def greedy_pick(action_value_dict, find_one=True, thresh=0):
        if find_one:  # find one maximizer
            return max(action_value_dict, key=action_value_dict.get)
        else:
            best_score = -float('inf')
            best_actions = []
            for action, value in action_value_dict.items():
                score = value.average if isinstance(value, utils.Averager) else value
                flag = utils.compare(score, best_score, thresh)
                if flag is 1:  # a strictly better action is found
                    best_score = score
                    best_actions = [action]
                elif flag is 0:  # an action which ties the best action is found
                    best_actions.append(action)
            return best_actions

    def choose_action(self, state, q_fun, off_policy=False):
        game = self.game
        explored = False
        if binomial(1, self.epsilon) == 1:
            action_chosen = choice(game.available_actions(state))
            explored = True
        else:
            action_chosen = choice(self.greedy_pick(q_fun[state], find_one=False))
        if off_policy:
            best_action = choice(self.greedy_pick(q_fun[state], find_one=False)) if explored else action_chosen
            for action in self.game.available_actions(state):
                if action == best_action:
                    self.prob_b[state][action] = 1 - self.epsilon + self.epsilon / len(game.available_actions(state))
                else:
                    self.prob_b[state][action] = self.epsilon / len(game.available_actions(state))
        return action_chosen

    def episode_generator(self, q_fun, state0=None, policy=None, off_policy=False):
        game = self.game
        state = game.reset(state0)
        state_ls = []
        action_ls = []
        reward_ls = []
        is_terminal = False
        while not is_terminal:
            state_ls.append(state)
            action = policy[state] if policy else self.choose_action(state, q_fun, off_policy)
            action_ls.append(action)
            state, reward, is_terminal = game.one_move(action)
            reward_ls.append(reward)
        return state_ls, action_ls, reward_ls

    def on_policy_mc_exploring_start(self, n_episodes=5*10**5, initial_policy=None):
        self.epsilon = 0
        game = self.game
        q_fun = game.q_initializer()
        for episode in range(n_episodes):
            state0 = game.reset()
            action0 = choice(game.available_actions(state0))
            state1, reward0, is_terminal = game.one_move(action0)
            state_ls, action_ls, reward_ls = [state0], [action0], [reward0]
            if not is_terminal:
                policy = None if episode else initial_policy
                state_ls_2, action_ls_2, reward_ls_2 = self.episode_generator(q_fun, state1, policy)
                state_ls.extend(state_ls_2)
                action_ls.extend(action_ls_2)
                reward_ls.extend(reward_ls_2)
            g = 0
            for s, a, r in zip(reversed(state_ls), reversed(action_ls), reversed(reward_ls)):
                g = game.gamma * g + r
                # using every-visit
                q_fun[s][a].add_new(g)
        return q_fun

    def on_policy_mc_epsilon_soft(self, n_episodes=5*10**5):
        game = self.game
        q_fun = game.q_initializer()
        for _ in range(n_episodes):
            state0 = game.reset()
            state_ls, action_ls, reward_ls = self.episode_generator(q_fun, state0)
            g = 0
            for s, a, r in zip(reversed(state_ls), reversed(action_ls), reversed(reward_ls)):
                g = game.gamma * g + r
                # using every-visit
                q_fun[s][a].add_new(g)
        return q_fun

    def off_policy_mc(self, n_episodes=5*10*5):
        game = self.game
        q_fun = {state: {action: 0 for action in game.available_actions(state)}
                 for state in game.state_space}
        c_fun = {state: {action: 0 for action in game.available_actions(state)}
                 for state in game.state_space}
        for _ in range(n_episodes):
            state0 = game.reset()
            state_ls, action_ls, reward_ls = self.episode_generator(q_fun, state0, off_policy=True)
            g = 0
            w = 1
            for s, a, r in zip(reversed(state_ls), reversed(action_ls), reversed(reward_ls)):
                g = game.gamma * g + r
                c_fun[s][a] += w
                q_fun[s][a] += w / c_fun[s][a] * (g - q_fun[s][a])
                best_action = choice(self.greedy_pick(q_fun[s], find_one=False))
                if best_action != a:
                    break
                w *= 1 / self.prob_b[s][a]
        return q_fun


if __name__ == '__main__':

    #
    game1 = BlackJack()
    agent1 = MC(game1)

    result_usable_ace = zeros((10, 13))
    result_no_usable_ace = zeros((10, 13))
    n_test = 10
    n_episodes = 5*10**5

    for _ in tqdm(range(n_test)):
        q1 = agent1.on_policy_mc_exploring_start(n_episodes=n_episodes, initial_policy=game1.policy_initializer())
        # q1 = agent1.on_policy_mc_epsilon_soft(n_episodes=n_episodes)
        # q1 = agent1.off_policy_mc(n_episodes=n_episodes)

        actions_usable_ace = zeros((10, 13))
        actions_no_usable_ace = zeros((10, 13))
        for key, value in q1.items():
            i = key[1] - 12
            if key[0] is 'A':
                j = 0
            elif key[0] is 'T':
                j = 9
            elif key[0] is 'J':
                j = 10
            elif key[0] is 'Q':
                j = 11
            elif key[0] is 'K':
                j = 12
            else:
                j = int(key[0]) - 1
            score_hit = value['hit'].average if isinstance(value['hit'], utils.Averager) else value['hit']
            score_stand = value['stand'].average if isinstance(value['stand'], utils.Averager) else value['stand']
            if key[2]:
                actions_usable_ace[i, j] = 1 if utils.compare(score_hit, score_stand, 0) is -1 else 0
            else:
                actions_no_usable_ace[i, j] = 1 if utils.compare(score_hit, score_stand, 0) is -1 else 0
        result_usable_ace += actions_usable_ace
        result_no_usable_ace += actions_no_usable_ace
    result_usable_ace /= n_test  # try to compare with RL book Fig 5.2(a)
    result_no_usable_ace /= n_test  # try to compare with RL book Fig 5.2(c)

    print('haha')
    #
