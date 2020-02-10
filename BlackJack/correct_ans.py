import pandas as pd
from tabular_learning import Agent
from game.blackjack import BlackJack

def make_pi_star():
    """Only have no usable A cases"""
    state_space = [(rank, i, False) for rank in "23456789KA" for i in range(11, 22)]
    pi = {}
    for state in state_space:
        rank, i, _ = state
        if rank in "23":
            pi[state] = 'stand' if i >= 13 else 'hit'
        elif rank in '456':
            pi[state] = 'stand' if i >= 12 else 'hit'
        elif rank in '789KA':
            pi[state] = 'stand' if i >= 17 else 'hit'
        else:
            raise Exception
    return pi

def get_q_star(soi=None, n=10**5):
    """compute Q*(S, A) for selected states S
    :param soi -- states of interest, e.g. [("K", 16, False), ("K", 17, False)...]
    if None, include all states in pi_star"""
    player = Agent(BlackJack(), n)
    pi_star = make_pi_star()
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
