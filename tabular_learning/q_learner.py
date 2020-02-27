from tabular_learning import Agent
import numpy as np
from tqdm import tqdm
from collections import defaultdict


class QLearner(Agent):
    """q_learning, sarsa, expected sarsa"""

    def __init__(self, game, alpha=10 ** -2, epsilon=0.1, n_trials=5 * 10 ** 4):
        super().__init__(game)
        self.alpha = alpha
        self.epsilon = epsilon

    def epsilon_greedy(self, state, q):
        if np.random.random() < self.epsilon:
            chosen_action = np.random.choice(self.game.available_actions(state))
        else:
            chosen_action = np.random.choice(self.greedy_action(q[state], find_one=False))
        return chosen_action

    def crude_run(self, episodes=5 * 10 ** 5, q0=None, soi=None):
        game = self.game
        q = game.q_initializer() if q0 is None else q0
        for _ in tqdm(range(episodes)):
            if soi is None:
                state = game.reset()
            else:
                state = game.reset(np.random.choice(soi))
            is_terminal = False
            while not is_terminal:
                action = self.epsilon_greedy(state, q)
                next_state, reward, is_terminal = game.one_move(action)
                if is_terminal:
                    q[state][action] += self.alpha * (reward - q[state][action])
                    break
                q_next = max(q[next_state].values())
                q[state][action] += self.alpha * (reward + game.gamma * q_next - q[state][action])
                state = next_state
        return q


    def q_car(self, episodes=5 * 10 ** 5, q0=None, soi=None, days = 30):
        # for jack_car_rent
        game = self.game
        q = game.q_initializer() if q0 is None else q0
        for _ in tqdm(range(episodes)):
            if soi is None:
                state = game.reset()
            else:
                state = game.reset(np.random.choice(soi))
            is_terminal = False
            while not is_terminal:
                for _ in range(days):
                    action = self.epsilon_greedy(state, q)
                    next_state, reward, is_terminal = game.one_move(action)
                    q_next = max(q[next_state].values())
                    q[state][action] += self.alpha * (reward + game.gamma * q_next - q[state][action])
                    state = next_state
                is_terminal = True
        return q


    def sarsa(self, episodes=50 * 10 ** 5, q0=None, soi=None):
        game = self.game
        q = game.q_initializer() if q0 is None else q0
        for _ in tqdm(range(episodes)):
            if soi is None:
                state = game.reset()
            else:
                state = game.reset(np.random.choice(soi))
            is_terminal = False
            action = self.epsilon_greedy(state, q)
            while not is_terminal:
                next_state, reward, is_terminal = game.one_move(action)
                if is_terminal:
                    q[state][action] += self.alpha * (reward - q[state][action])
                    break
                next_action = self.epsilon_greedy(next_state, q)
                q[state][action] += self.alpha * (reward + game.gamma
                                                          * q[next_state][next_action] - q[state][action])
                state = next_state
                action = next_action
        return q

    def expected_sarsa(self, episodes= 50 * 10 **5, q0=None, soi=None):
        game = self.game
        q = game.q_initializer() if q0 is None else q0
        for _ in tqdm(range(episodes)):
            if soi is None:
                state = game.reset()
            else:
                state = game.reset(np.random.choice(soi))
            is_terminal = False
            action = self.epsilon_greedy(state, q)
            while not is_terminal:
                next_state, reward, is_terminal = game.one_move(action)
                if is_terminal:
                    q[state][action] += self.alpha * (reward - q[state][action])
                    break
                next_action = self.epsilon_greedy(next_state, q)
                candidate_lst = list(v for v in q[next_state].values()) + [q[next_state][next_action]]
                prob_dist = len(q[next_state]) * [self.epsilon] + \
                            [(len(q[next_state]) * (1 - self.epsilon))]
                prob_list = [prob / sum(prob_dist) for prob in prob_dist]
                target = np.dot(candidate_lst, prob_list)
                q[state][action] += self.alpha * (reward + game.gamma * target - q[state][action])
                state = next_state
                action = next_action
        return q


    def afterstate_epi_greedy(self, q, state):
        game = self.game
        if np.random.random() < self.epsilon:
            chosen_action = np.random.choice(game.available_actions(state))
            after_state = game.half_move(chosen_action, state)
            next_state, reward, is_terminal = game.one_move(chosen_action)
            return after_state, next_state, reward, is_terminal
        else:
            after_state = self.best_after_state(q, state)
            chosen_action = game.half_move_reverse(after_state, state)
            next_state, reward, is_terminal = game.one_move(chosen_action)

            return after_state, next_state, reward, is_terminal

    def best_after_state(self, q, state):
        game = self.game
        afterstate_lst = [game.half_move(action, state) for action in game.available_actions(state)]
        candidate_q = {afterstate : q[afterstate] for afterstate in afterstate_lst}
        max_q = max(candidate_q.values())
        candidate_afterstate = [key for key, value in candidate_q.items() if value==max_q]
        num = np.random.choice(range(len(candidate_afterstate)))
        return candidate_afterstate[num]

    def afterstates_q_init(self):
        game = self.game
        if game.state_space is None:
            return defaultdict(int)
        else:
            q = {k:0 for k in game.state_space}
            return q

    def afterstate_q(self, episodes=5*10**6):
        game = self.game
        q = self.afterstates_q_init()
        for _ in tqdm(range(episodes)):
            state = game.reset()
            is_terminal = False
            while not is_terminal:
                after_state, next_state, reward, is_terminal = self.afterstate_epi_greedy(q, state)
                if is_terminal:
                    q[after_state] += self.alpha * (reward - q[after_state])
                    break
                next_after_state = self.best_after_state(q, next_state)
                target = q[next_after_state]
                q[after_state] += self.alpha * (reward + game.gamma * target - q[after_state])
                state = next_state
        return q

    def afterstate_q_car(self, episodes=5*10**6, length = 14):
        # for car_rent-like cases
        game = self.game
        q = self.afterstates_q_init()
        for _ in tqdm(range(episodes)):
            state = game.reset()
            for i in range(length):
                after_state, next_state, reward, is_terminal = self.afterstate_epi_greedy(q, state)
                next_after_state = self.best_after_state(q, after_state)
                target = q[next_after_state]
                q[after_state] += self.alpha * (reward + game.gamma * target - q[after_state])
                state = next_state
        return q

    def afterstate_sarsa(self, episodes=50*10**5):
        game = self.game
        q = self.afterstates_q_init()
        for i in tqdm(range(episodes)):
            state = game.reset()
            is_terminal = False
            while not is_terminal:
                after_state, next_state, reward, is_terminal = self.afterstate_epi_greedy(q, state)
                if is_terminal:
                    q[after_state] += self.alpha * (reward - q[after_state])
                    break
                next_after_state, _, _, _ = self.afterstate_epi_greedy(q, next_state)
                target = q[next_after_state]
                q[after_state] += self.alpha * (reward + game.gamma * target - q[after_state])
                state = next_state
        return q

    def afterstate_sarsa_car(self, episodes=50*10**5, length =14):
        game = self.game
        q = self.afterstates_q_init()
        for i in tqdm(range(episodes)):
            state = game.reset()
            for j in range(length):
                after_state, next_state, reward, _ = self.afterstate_epi_greedy(q, state)
                next_after_state, _, _, _ = self.afterstate_epi_greedy(q, next_state)
                target = q[next_after_state]
                q[after_state] += self.alpha * (reward + game.gamma * target - q[after_state])
                state = next_state
        return q

if __name__ == '__main__':
    from game.jacks_car_rental import JacksCarRental
    test_q = QLearner(JacksCarRental())
    q1 = test_q.afterstate_q_car(5 * 10 ** 4)
    value = np.zeros((21,21))
    for key, v in q1.items():
        value[key[0]][key[1]] = v

