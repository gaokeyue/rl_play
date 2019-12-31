from functools import partial
from operator import itemgetter
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
plt.style.use('seaborn')

class PMF:


    def __init__(self, prob):
        """probability mass function, pmf[x] = p
        :param prob (dict) -- prob[k] = Pr{X = k}"""
        self.prob = prob

    def expectation(self, f=None):
        if f is None:
            return sum(k * p for k, p in self.prob.items())
        return sum(f[k] * p for k, p in self.prob.items())


class State:
    pass

class Action:
    pass

class Policy:
    pass


class RLDP:
    def __init__(self, dynamics, decay=1):
        """Suppose p(s',r|s, a) is known
        :param dynamics (dict) -- {state: {action: {state_prob: pmf, reward_prob: pmf}}}
        """
        self.dynamics = dynamics
        self.decay = decay

    def policy_iter(self, policy_initializer):
        pi0 = policy_initializer()
        v0 = self.policy_eval(pi0)
        max_iter = 100
        n_iter = 0
        while n_iter < max_iter:
            n_iter += 1

            pi1 = self.policy_impro(v0)

    def state_action_value_one(self, state, action, state_values):
        """"""
        a = self.dynamics[state][action]['reward'].expectation()
        b = self.decay * self.dynamics[state][action]['state'].expectation(state_values)
        return a + b


    def policy_eval(self):
        pass

    def policy_impro(self, state_values):
        pi = {}
        for state in self.state_space:
            actions = self.dynamics[state].keys()
            best_actions = set()
            best_score = -np.inf
            for action in actions:
                q = self.state_action_value_one(state, action, state_values)
                if q == best_score:
                    best_actions.add(action)
                elif q > best_score:
                    best_actions = {action}
                    best_score = q
        pass

    def value_iter(self, value_initialzer):
        v0 = value_initialzer()
        v1 = {}
        for state, actions in self.dynamics.items():
            # max_a q(s, a)
            v1[state] = max(self.state_action_value_one(state, action, v0) for action in actions)



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

class Gambler:
    def __init__(self, p_h, goal=100):
        """:param p_h -- the probability of head
        :param goal -- once goal is reached, the game ends and receives reward 1.
        """
        self.p_h = p_h
        self.goal = goal
        # two terminal states, 0 and goal, included
        self.state_space = tuple(range(0, goal + 1))
        # action_space[i] is the max stakes when state = i
        # For the two terminal states, do nothing, i.e bid 0
        self.action_space = [state if state * 2 <= goal else goal - state
                             for state in self.state_space]
        self.eps = 10 ** -6

    def is_terminal(self, state):
        return state <= 0 or state >= self.goal

    def get_policy_dynamics(self, policy):
        """Once a policy pi is specified, the expectation of immediate reward
        and the state transition probability under pi are both determined.
        :param policy -- deterministic map state to action (stake)
        """
        # The immediate reward. Nonzero only if state + action is goal
        reward_pi = np.zeros(self.goal + 1)
        state_tran_pi = np.zeros((self.goal + 1, self.goal + 1))
        for state, action in enumerate(policy):
            if self.is_terminal(state):
                # the terminal state can no longer generate positive reward
                state_tran_pi[state, state] = 1
            else:
                state_tran_pi[state, state - action] += 1 - self.p_h  # use += in case action=0
                try:
                    state_tran_pi[state, state + action] += self.p_h
                except:
                    print('haha')
                if state + action == self.goal:
                    reward_pi[state] = self.p_h
        return reward_pi, state_tran_pi

    def policy_eval(self, policy, v0=None):
        """v(s) = sum(p(s'|s)v(s')) + E_pi[r]"""
        # initialize value function
        if v0 is None:
            v0 = .5 * np.ones_like(self.state_space)
            v0[0] = 0
            v0[-1] = 0
        reward_pi, state_tran_pi = self.get_policy_dynamics(policy)
        max_iter = 100
        n_iter = 0
        while n_iter < max_iter:
            n_iter += 1
            v1 = reward_pi + state_tran_pi @ v0
            if np.linalg.norm(v1 - v0, np.inf) <= self.eps:
                break
            v0 = v1
        else:
            print(f"exceed {max_iter}")
        return v0

    def policy_impro(self, state_val):

        def state_action_value(state, action):
            """Assume that state is non terminal"""
            reward = self.p_h if state + action == self.goal else 0
            # the expectation of the value function of s'
            val_exp = (1 - self.p_h) * state_val[state - action] \
                      + self.p_h * state_val[state + action]
            return reward + val_exp

        pi = np.zeros_like(self.state_space)
        for state in self.state_space:
            if not self.is_terminal(state):
                actions = range(1, 1 + self.action_space[state])
                a_star = max(actions, key=partial(state_action_value, state))
                pi[state] = a_star
        return pi

    def policy_iter(self, pi0=None):
        if pi0 is None:
            # the initial policy is to bet as much as one can
            pi0 = np.array(self.action_space)
        max_iter = 100
        n_iter = 0
        while n_iter < max_iter:
            n_iter += 1
            print(pi0)
            state_val = self.policy_eval(pi0)
            pi1 = gambler.policy_impro(state_val)
            if all(pi0[state] == pi1[state] for state in self.state_space):
                break
            pi0 = pi1
        else:
            print(f"exceeds maximum number of iteration {max_iter}")
        return pi0

    def value_iter(self):
        pass

if __name__ == '__main__':
    gambler = Gambler(p_h=.4, goal=100)
    opt_pi = gambler.policy_iter()
    plt.plot(opt_pi)
    plt.show()