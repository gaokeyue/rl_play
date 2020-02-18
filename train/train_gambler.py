import os
import pickle
import pandas as pd
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from utils import expectation
from game.gambler import Gambler
from tabular_learning.q_learner_plus import QLearner
import tabular_learning.dynamic_prog as dp


def get_q_star(goal, p_h=.4):
    gambler_dp = dp.Gambler(goal=goal, p_h=p_h, eps=10 ** -8)
    pi_star = gambler_dp.policy_initializer()
    v_star = gambler_dp.policy_eval(pi_star, v0=gambler_dp.state_value_initializer(), max_iter=10 ** 3)
    q_star = {}

    for state in gambler_dp.state_space:
        act_val_dct = {}
        for action in range(1, min(state, gambler_dp.goal - state) + 1):
            reward_prob, state_prob = gambler_dp.dynamics[state][action]
            act_val_dct[action] = expectation(reward_prob) + expectation(state_prob, v_star)
        q_star[state] = act_val_dct
    return q_star


def q_learning(goal, p_h=.4):
    Learner = QLearner(Gambler(goal=goal, p_h=p_h))
    q_hat = Learner.q_learning(n_episodes=10 ** 7)
    print(q_hat)
    ss = pd.DataFrame.from_dict(q_hat, orient='index').stack()
    ss.sort_index(inplace=True)
    ss.index.names = ['state', 'action']
    ss.name = 'value'
    # ss.to_frame().to_csv('~/Desktop/gambler_q.csv')
    return ss


def make_model():
    model = Sequential()
    model.add(Dense(units=4, activation='relu', input_dim=2))
    model.add(Dense(units=1))
    model.compile(optimizer=SGD(lr=10**-3), loss='mse')
    return model

def greedy_pick(game, state, model):
    actions = game.available_actions(state)
    values = model.predict(np.array([[state, action] for action in actions])).flatten()
    return max(zip(values, actions))

if __name__ == '__main__':
    project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_dir = os.path.join(project_dir, 'data')
    goal = 15
    game = Gambler(goal=goal)
    model = make_model()
    n_episodes = 10 ** 3
    eps = .1
    for i in range(n_episodes):
        if i % 50 == 0:
            print(f"Running the {i}th episode==================")
        state0 = game.reset()
        # action0 = np.random.choice(game.available_actions())
        _, action0 = greedy_pick(game, state0, model)
        while True:
            state1, reward, done = game.one_move(action0)
            if done:
                x = np.array([[state0, action0]])
                y = np.array([reward])
                model.fit(x, y, verbose=0)
                break
            else:
                actions = game.available_actions(state1)
                if np.random.random() < eps:
                    action1 = np.random.choice(actions)
                else:
                    action1, v = greedy_pick(game, state1, model)
                    x = np.array([[state0, action0]])
                    y = np.array([reward + game.gamma*v])
                    model.fit(x, y, verbose=0)



