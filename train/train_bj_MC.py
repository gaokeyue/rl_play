import os
import pickle
import pandas as pd
from tabular_learning.mc_controller import MC
from game import BlackJack


def target_policy():
    state_space = [(dealer, hand_sum, usable) for dealer in "23456789TJQKA"
                   for hand_sum in range(11, 22) for usable in [True, False]]
    pi = {}
    for state in state_space:
        dealer, hand_sum, usable = state
        if usable:
            if dealer in "2345678":
                pi[state] = 'stand' if hand_sum >= 18 else 'hit'
            elif dealer in "9TJQKA":
                pi[state] = 'stand' if hand_sum >= 19 else 'hit'
        else:
            if dealer in "23":
                pi[state] = 'stand' if hand_sum >= 13 else 'hit'
            elif dealer in '456':
                pi[state] = 'stand' if hand_sum >= 12 else 'hit'
            elif dealer in '789TJQKA':
                pi[state] = 'stand' if hand_sum >= 17 else 'hit'
    return pi


if __name__ == '__main__':

    print('haha')
    xx = target_policy()
    project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_dir = os.path.join(project_dir, 'data')
    game = BlackJack()
    player = MC(game)
    # crude run
    q0 = None
    n_episodes = 10**6
    for thresh in [.5, .1, .05]:
        q0 = player.on_policy_mc_exploring_start(n_episodes=n_episodes, policy0=game.policy_initializer(), q0=q0)
        # q0 = player.on_policy_mc_epsilon_soft(n_episodes=n_episodes, q0=q0)
        # q0 = player.off_policy_mc(n_episodes=n_episodes, q0=q0)
        q0 = player.action_prune(q0, thresh)
    # fine tune
    pi_star = player.fine_tune(q0)
    print(player.compare_dict(pi_star, target_policy()))
