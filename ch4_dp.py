import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
from utils import deco_timer, deco_while

def expectation(prob_dict, f=None):
    if f is None:  # just the expectation of the probability distribution
        return sum(x * p for x, p in prob_dict.items())
    # if f is given, then E[f(X)]
    return sum(f[x] * p for x, p in prob_dict.items())

def compare(a, b, thresh):
    c = a - b
    if c > thresh:
        return 1
    if c < -thresh:
        return -1
    return 0

def allmax(input_dict, thresh):

    best_score = -np.inf
    best_keys = []
    for k, v in input_dict.items():
        flag = compare(v, best_score, thresh)
        if flag is 1:
            best_score = v
            best_keys = [k]
        elif flag is 0:
            best_keys.append(k)
    return best_keys, best_score


class DPRL:
    def __init__(self, dynamics, gamma=1, eps=10**-8, max_iter=1000):
        """The four argument p(s',r|s,a) completely specifies the dynamics.
        :param dynamics (dict of dict of tuple of dict) --
        dynamics[state][action] = (reward_prob, state_prob)
        where foo_prob is discrete probability distribution, given by dictionary
        foo_prob[k] = Pr{X=k}
        :param gamma (0 < float <= 1) -- decay rate
        :param eps (float) --
        """
        self.dynamics = dynamics
        self.num_states = len(dynamics)
        self.gamma = gamma
        self.eps = eps
        self.max_iter = max_iter

    def greedy_one_state(self, state_values, state):
        action_value_dict = {}
        best_actions = []
        best_score = -np.inf
        for action, (reward_prob, state_prob) in self.dynamics[state].items():
            r = expectation(reward_prob) + \
                self.gamma * expectation(state_prob, state_values)
            action_value_dict[action] = r
            flag = compare(r, best_score, self.eps)
            if flag is 1:
                best_score = r
                best_actions = [action]
            elif flag is 0:
                best_actions.append(action)
        return best_score, best_actions, action_value_dict

    def value_iter(self, v0):
        """"""
        @deco_while(max_iter=self.max_iter, print_iter=10, thresh=self.eps)
        def value_iteration_step(v0):
            # print(v0)
            v1 = np.zeros(self.num_states)
            for state in range(self.num_states):
                # action_value_dict = self.get_state_action_value(v0, state)
                # _, best_v = allmax(action_value_dict, self.eps)
                # pick the best action with repsect to the current value function
                actions = self.dynamics[state]
                best_v = max(expectation(reward_prob) +
                                self.gamma * expectation(state_prob, v0)
                                for reward_prob, state_prob in actions.values())
                v1[state] = best_v
            return v1

        result = value_iteration_step(v0)
        # max_iter = 1000
        # n_iter = 0
        # while n_iter < max_iter:
        #     n_iter += 1
        #     if n_iter % 10 == 0:
        #         print(f"Running iteration {n_iter}")
        #     v1 = step(v0)
        #     if all(abs(x - y) <= self.eps for x, y in zip(v0, v1)):
        #         break
        #     v0 = v1
        # else:
        #     print(f"exceed maximum iteration {max_iter}")
        return result

    def policy_iter(self, pi0, v0):
        max_iter = 1000
        n_iter = 0
        while n_iter < max_iter:
            n_iter += 1
            v0 = self.policy_eval(pi0, v0)
            v1, pi1 = self.policy_improve(v0)
            if all(abs(x - y) <= self.eps for x, y in zip(v0, v1)):
                v0 = v1
                pi0 = pi1
                break
            for state in pi0:
                pi0[state] = pi1[state].pop()
        else:
            print(f"exceed maximum iteration {max_iter}")
        return v0, pi0

    def get_trans_prob_pi(self, policy):
        """return transition probability under a deterministic policy pi
        :param policy -- deterministic map state to action (stake)
        """
        reward_pi = np.zeros(self.num_states)
        state_tran_pi = np.zeros((self.num_states, self.num_states))
        for state in range(self.num_states):
            action = policy[state]
            reward_prob, state_prob = self.dynamics[state][action]
            reward_pi[state] = expectation(reward_prob)
            for s_p, p in state_prob.items():
                state_tran_pi[state, s_p] = p
        return reward_pi, state_tran_pi

    def policy_eval(self, pi, v0):
        reward_pi, state_tran_pi = self.get_trans_prob_pi(pi)
        max_iter = 1000
        n_iter = 0
        while n_iter < max_iter:
            n_iter += 1
            v1 = reward_pi + self.gamma * state_tran_pi @ v0
            if all(abs(x - y) <= self.eps for x, y in zip(v0, v1)):
                break
            v0 = v1
        return v0

    def policy_improve(self, v0):
        v1 = np.zeros(self.num_states)
        pi1 = {}
        for state, value in enumerate(v0):
            new_value, best_actions, _ = self.greedy_one_state(v0, state)
            v1[state] = new_value
            pi1[state] = best_actions
        return v1, pi1


class JackCarRental:
    def __init__(self):
        self.eps = 10 ** -4
        self.gamma = 0.9
        self.max_capacity = 20
        self.max_trans = 5
        self.rev = 10
        self.cost = 2
        self.la_pick_x = 3
        self.la_drop_x = 3
        self.la_pick_y = 4
        self.la_drop_y = 2
        self.poisson_table = {la: [poisson(la, k) for k in range(21)] for la in
                              {self.la_pick_x, self.la_pick_y, self.la_drop_x, self.la_drop_y}}
    pass

    def policy_eval(self, pi):
        """pi(state) = action. action(state) = state'
        :param pi (dict)  -- pi[state] = action
        p(s', r|s, a)
        """
        # initialize value function
        v0 = np.ones((self.max_capacity+1, self.max_capacity+1))
        n_iter = 0
        max_iter = 100
        while n_iter < max_iter:
            if n_iter % 1 == 0:
                print(f"This is iteration {n_iter}")
            v1 = self.step(v0, pi)
            if abs(v1 - v0).max() < self.eps:
                break
            v0 = v1
        else:
            print(f"exceed max iterations {max_iter}")

    def step(self, v0, pi):
        v1 = np.zeros((self.max_capacity+1, self.max_capacity+1))
        for x in range(self.max_capacity + 1):
            for y in range(self.max_capacity + 1):
                print(f"updating state {(x, y)}")
                action = pi[x, y]
                s_morning = (x - action, y + action)
                v1[x, y] = self.morning_dynamic(s_morning, v0) - action * self.cost

    def morning_dynamic(self, s, v0):
        """n_pick_x, pick_y, drop_x, drop_y"""

        x0, y0 = s
        r = 0
        for n_pick_x in range(self.max_capacity+1):
            for n_drop_x in range(self.max_capacity+1):
                for n_pick_y in range(self.max_capacity+1):
                    for n_drop_y in range(self.max_capacity+1):
                        prob_pick_x = self.poisson_table[self.la_pick_x][n_pick_x]
                        prob_drop_x = self.poisson_table[self.la_drop_x][n_drop_y]
                        prob_pick_y = self.poisson_table[self.la_pick_y][n_pick_y]
                        prob_drop_y = self.poisson_table[self.la_drop_y][n_drop_y]
                        prob = prob_pick_x * prob_drop_x * prob_pick_y * prob_drop_y
                        if n_pick_x > x0:
                            n_pick_x = x0
                        if n_pick_y > y0:
                            n_pick_y = y0
                        r += prob * self.rev * (n_pick_x + n_pick_y)
                        x1 = x0 - n_pick_x + n_drop_x
                        y1 = y0 - n_pick_y + n_drop_y
                        if x1 > self.max_capacity:
                            x1 = self.max_capacity
                        if y1 > self.max_capacity:
                            y1 = self.max_capacity
                        r += prob * self.gamma * v0[x1, y1]
        return r


class Gambler(DPRL):
    def __init__(self, p_h, goal=100, eps=10**-6, max_iter=1000):
        self.p_h = p_h
        self.goal = goal
        self.state_space = range(self.goal + 1)
        dynamics = self.build_dynamics(p_h, goal)
        super().__init__(dynamics, gamma=1, eps=eps, max_iter=max_iter)

    def build_dynamics(self, p_h, goal):
        dynamics = {}
        for state in self.state_space:
            if self.is_terminal(state):
                reward_prob = {0: 1}
                state_prob = {state: 1}
                actions = {0: (reward_prob, state_prob)}
            else:
                actions = {}
                # the gambler at least bids 1
                for action in range(1, min(state, goal - state) + 1):
                    if state + action == goal:
                        # If there is a chance to win, then get reward 1 with prob p_h
                        reward_prob = {0: 1 - p_h, 1: p_h}
                    else:
                        reward_prob = {0: 1}
                    state_prob = {state - action: 1 - p_h,
                                  state + action: p_h}
                    actions[action] = (reward_prob, state_prob)
            dynamics[state] = actions
        return dynamics

    def is_terminal(self, state):
        return state <= 0 or state >= self.goal

    def state_value_initializer(self):
        v0 = np.linspace(0, 1, self.num_states)
        # v0[0] = 0
        v0[-1] = 0
        return v0

    def policy_initializer(self):
        return {state: min(state, self.goal - state)
                for state in self.state_space}

@deco_timer
def test_gambling(p_h=.6, goal=15, eps=10**-6, max_iter=7000):
    gambler = Gambler(p_h, goal, eps=eps, max_iter=max_iter)
    # v0 = gambler.state_value_initializer()
    # pi0 = gambler.policy_initializer()
    # v_star, pi_star = gambler.value_iter(v0)
    v_star = gambler.value_iter(gambler.state_value_initializer())
    print(v_star)
    _, (v_plot, pi_plot) = plt.subplots(2, 1)
    v_plot.plot(v_star, '.')
    v_plot.set_title('optimal value function')
    v_plot.set_xlabel('state')
    v_plot.set_ylabel('v_star')
    xx = []
    yy = []
    for state in gambler.state_space:
        best_v, best_actions, action_value_dict = gambler.greedy_one_state(v_star, state)
        for action in best_actions:
            xx.append(state)
            yy.append(action)
    pi_plot.set_title('optimal policy family')
    pi_plot.plot(xx, yy, '.')
    pi_plot.set_xlabel('state')
    pi_plot.set_ylabel('optimal action')
    plt.show()
    # return v_star, pi_star

if __name__ == '__main__':
    # gambler = Gambler(p_h=.4, goal=64, eps=10**-8)
    # # v0 = gambler.state_value_initializer()
    # # pi0 = gambler.policy_initializer()
    # # v1, pi_star = gambler.policy_iter(pi0, v0)
    # # pi_vec = np.zeros_like(v0)
    # # for state, actions in pi_star.items():
    # #     pi_vec[state] = actions[0]
    # # plt.plot(pi_vec, 'o''')
    # v0 = gambler.state_value_initializer()
    # v_star = gambler.value_iter(v0)
    # xx = []
    # yy = []
    # # pi_star = np.zeros_like(v_star)
    # pi_star = {}
    # for state in gambler.state_space:
    #     best_v, best_actions, action_value_dict = gambler.greedy_one_state(v_star, state)
    #     for action in best_actions:
    #         xx.append(state)
    #         yy.append(action)
    #     # print(action_value_dict)
    #     # if len(best_actions) > 1:
    #     #     print(f"The best actions for state {state} are {best_actions}")
    #     pi_star[state] = best_actions
    # plt.plot(xx, yy, '.')
    # # plt.plot(pi_star, '.')
    # # print(v_star)
    # # plt.plot(v_star, '.')
    # plt.xlabel('state')
    # plt.ylabel('optimal action')
    # plt.grid()
    # plt.show()
    test_gambling()
