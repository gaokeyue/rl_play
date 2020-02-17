import os
import pickle
import pandas as pd
from tabular_learning.q_learner_plus import QLearner
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

def get_q_star(soi=None, n=10**5):
    """compute Q*(S, A) for selected states S
    :param soi -- states of interest, e.g. [("K", 16, False), ("K", 17, False)...]
    if None, include all states in pi_star"""
    player = QLearner(BlackJack())
    pi_star = target_policy()
    if soi is not None:
        soi = {state: ['hit', 'stand'] for state in soi}
    q_star = player.policy_eval(pi_star, soi, n)
    return q_star



if __name__ == '__main__':
    print('haha')
    # n = 5 * 10 ** 3
    xx = target_policy()
    project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_dir = os.path.join(project_dir, 'data')
    player = QLearner(BlackJack())
    # crude run
    q0 = None
    alpha = 10**-3
    for thresh in [.5, .1, .05]:
        q0 = player.q_learning(q0=q0, alpha=10**-3, n_episodes=10**7)
        q0 = player.action_prune(q0, thresh)
        alpha *= .5
    # fine tune
    pi_star = player.fine_tune(q0)
    player.compare_dict(pi_star, target_policy())

