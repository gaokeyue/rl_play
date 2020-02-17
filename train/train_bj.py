import pandas as pd
from tabular_learning import Agent
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
        if dealer in "23":
            pi[state] = 'stand' if hand_sum >= 13 else 'hit'
        elif dealer in '456':
            pi[state] = 'stand' if hand_sum >= 12 else 'hit'
        elif dealer in '789KA':
            pi[state] = 'stand' if hand_sum >= 17 else 'hit'
    return pi

def get_q_star(soi=None, n=10**5):
    """compute Q*(S, A) for selected states S
    :param soi -- states of interest, e.g. [("K", 16, False), ("K", 17, False)...]
    if None, include all states in pi_star"""
    player = Agent(BlackJack(), n)
    pi_star = target_policy()
    if soi is not None:
        soi = {state: ['hit', 'stand'] for state in soi}
    q_star = player.policy_eval(pi_star, soi)
    return q_star

if __name__ == '__main__':
    print('haha')
    n = 5 * 10 ** 3
    q_star = get_q_star(n=n)
    df = pd.DataFrame(q_star).T
    print(df.head())
    df['better'] = df.apply(lambda row: 'hit' if row['hit'] > row['stand'] else 'stand', axis=1)
    # df.to_csv('~/Desktop/cankao.csv')
