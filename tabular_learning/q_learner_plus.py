import os
import random
from operator import itemgetter
from tqdm import tqdm
from utils import convex_comb
from tabular_learning import Agent


class QLearner(Agent):
    def q_learning(self, q0=None, alpha=10 ** -3, n_episodes=10 ** 6, eps=.1):
        """Q learning with actions possibly pruned.
        :param q0 (dict or None) -- Initial state-action value dictionary.
        If None, all state-action pairs are trained.
        :param alpha -- learning rate for q learning
        :param n_episodes -- number of episodes run
        :param eps -- epsilon greedy
        """
        game = self.game
        if q0 is None:
            q_dict = {}
            sa_oi = None
        else:
            q_dict = q0
            # state-action pair of interest, if state has more than one action in q0
            sa_oi = []
            for state, action_value_dict in q_dict.items():
                if len(action_value_dict) > 1:
                    for action in action_value_dict:
                      sa_oi.append((state, action))
        for _ in tqdm(range(n_episodes)):
            if sa_oi is None:  # first q learning procedure
                s0 = game.reset()
                if s0 in q_dict:
                    act_val0 = q_dict[s0]
                    a0, v0 = Agent.epsilon_greedy(act_val0, eps)
                else:
                    actions = game.available_actions()
                    a0 = random.choice(actions)
                    v0 = 1
                    act_val0 = {action: v0 for action in actions}
                    q_dict[s0] = act_val0
            else:
                s0, a0 = random.choice(sa_oi)
                game.reset(s0)
                act_val0 = q_dict[s0]  # s0 must be in q_dict since s0 is of interest
                v0 = act_val0[a0]
            while True:
                s1, r, is_terminal = game.one_move(a0)
                if is_terminal:
                    act_val0[a0] = convex_comb(v0, r, alpha)
                    break
                else:
                    if s1 in q_dict:
                        act_val1 = q_dict[s1]
                        a1, v1 = max(act_val1.items(), key=itemgetter(1))
                        act_val0[a0] = convex_comb(v0, r + game.gamma * v1, alpha)
                        act_val0 = act_val1
                        if random.random() < eps:
                            a0 = random.choice(list(act_val0))
                            v0 = act_val1[a0]
                        else:
                            a0, v0 = a1, v1
                    else:  # if next state not encounted before, no need for updating
                        actions = game.available_actions()
                        a0 = random.choice(actions)
                        v0 = 1
                        act_val0 = {action: v0 for action in game.available_actions()}
        return q_dict


if __name__ == '__main__':
    project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_dir = os.path.join(project_dir, 'data')
