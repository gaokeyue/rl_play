from tabular_learning import Agent
from game.blackjack import BlackJack
import random
import numpy as np


class MC(Agent):

    def on_policy_mc_es(self, n_episodes=5*10**5):
        game = self.game
        q_fun = game.q_initializer()
        for i in range(n_episodes):
            initial_state = game.reset()
            game._action = random.choice(game.available_actions())
            new_state, reward, is_terminal = game.one_move(game.action)
            state_ls, action_ls, reward_ls = [initial_state], [game.action], [reward]
            if i is 0:
                policy = game.policy_initializer()
            if not is_terminal:
                state_ls_2, action_ls_2, reward_ls_2 = self.policy_run(policy, game.state)
                state_ls.extend(state_ls_2)
                action_ls.extend(action_ls_2)
                reward_ls.extend(reward_ls_2)
            g = 0
            for s, a, r in zip(reversed(state_ls), reversed(action_ls), reversed(reward_ls)):
                g = game.gamma * g + r
                # using every-visit
                q_fun[s][a].add_new(g)
                best_action = random.choice(self.greedy_pick(q_fun[s], find_one=False, thresh=0))
                policy[s] = best_action
        return q_fun


if __name__ == '__main__':

    #
    game1 = BlackJack()
    agent1 = MC(game1)
    q1 = agent1.on_policy_mc_es()
    actions_usable_ace = np.zeros((10, 10))
    actions_no_usable_ace = np.zeros((10, 10))
    for key, value in q1.items():
        if key[2]:
            i = key[1] - 12
            j = 0 if key[0] is 'A' else (game1._cards_count[key[0]] - 1)
            actions_usable_ace[i, j] = 1 if value['hit'].average < value['stand'].average else 0
            # try to compare with RL book Fig 5.2(a)
        else:
            i = key[1] - 12
            j = 0 if key[0] is 'A' else (game1._cards_count[key[0]] - 1)
            actions_no_usable_ace[i, j] = 1 if value['hit'].average < value['stand'].average else 0
            # try to compare with RL book Fig 5.2(c)
    print('haha')
    #
