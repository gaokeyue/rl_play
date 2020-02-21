import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from game import Game

ALPHA = 0.5
EPSILON = 0.1
DISCOUNT = 0.9


class JacksCarRental(Game):
    def __init__(self):
        self.rent_price = 10
        self.move_cost = 2
        self.parking_cap = 10
        self.max_move = 5
        self.lambda_rent_loc1 = 3
        self.lambda_return_loc1 = 3
        self.lambda_rent_loc2 = 4
        self.lambda_return_loc2 = 2
        self._state = np.array([random.randint(0, self.parking_cap),
                                random.randint(0, self.parking_cap)])
        
    @property
    def state(self):
        return tuple(self._state)

    def reset(self):
        self._state = np.array([random.randint(0, self.parking_cap),
                                random.randint(0, self.parking_cap)])
        return tuple(self._state)

    def one_move(self, action):
        self._state[0] -= action
        self._state[1] += action

        reward = - action * self.move_cost

        cars_rented_loc1 = self._state[0] + 1
        while cars_rented_loc1 > self._state[0]:
            cars_rented_loc1 = max(0, np.random.poisson(self.lambda_rent_loc1, 1)[0])

        cars_rented_loc2 = self._state[1] + 1
        while cars_rented_loc2 > self._state[1]:
            cars_rented_loc2 = max(0, np.random.poisson(self.lambda_rent_loc2, 1)[0])

        self._state[0] -= cars_rented_loc1
        self._state[1] -= cars_rented_loc2

        self._state[0] = min(self.parking_cap, self.state[0] + np.random.poisson(self.lambda_return_loc1, 1)[0])
        self._state[1] = min(self.parking_cap, self.state[1] + np.random.poisson(self.lambda_return_loc2, 1)[0])

        reward += (cars_rented_loc1 + cars_rented_loc2) * self.rent_price
        return self.state, reward, False
    
    def available_actions(self, state=None):
        max_loc1_can_get = min(self.max_move, self.parking_cap - self.state[0])
        max_loc2_can_get = min(self.max_move, self.parking_cap - self.state[1])
        max_loc1_can_move = min(self.max_move, self.state[0])
        max_loc2_can_move = min(self.max_move, self.state[1])
        max_loc1to2 = min(max_loc1_can_move, max_loc2_can_get)
        max_loc2to1 = min(max_loc2_can_move, max_loc1_can_get)
        range_lower_bound = self.max_move - max_loc2to1
        range_upper_bound = self.max_move + max_loc1to2
        return range(range_lower_bound, range_upper_bound + 1)


class Sarsa:
    def __init__(self, environment):
        self.env = environment
        self.n = self.env.parking_cap + 1
        self.Q = np.zeros((self.n, self.n, self.n))

    def reset_q(self):
        self.Q = np.zeros((self.n, self.n, self.n))

    def choose_action(self):
        range_lower_bound = min(self.env.available_actions(self.env.state))
        range_upper_bound = max(self.env.available_actions(self.env.state))
        action_range = range_upper_bound - range_lower_bound

        if random.random() <= EPSILON:
            action = random.randint(0, action_range) + range_lower_bound
        else:
            action = np.argmax(self.Q[self.env.state[0], self.env.state[1],
                               range_lower_bound: range_upper_bound + 1]) + range_lower_bound

            # If multiple actions have the same max value, we need to choose one of them randomly
            available_actions = []
            for i in range(action_range + 1):
                if self.Q[self.env.state[0], self.env.state[1], i + range_lower_bound]\
                        == self.Q[self.env.state[0], self.env.state[1], action]:
                    available_actions.append(i + range_lower_bound)

            action = random.choice(available_actions)

        return action

    def learn(self, times):
        self.env.reset()

        reward = 0

        action = self.choose_action()

        for i in tqdm(range(times)):
            prev_state = np.copy(self.env.state)

            _, reward, _ = self.env.one_move(action - self.env.max_move)

            next_action = self.choose_action()

            self.Q[prev_state[0], prev_state[1], action] += \
                ALPHA * (reward + DISCOUNT * self.Q[self.env.state[0], self.env.state[1], next_action]
                         - self.Q[prev_state[0], prev_state[1], action])

            action = next_action

    def view_policy(self):
        policy = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                policy[i, j] = np.argmax(self.Q[i, j]) - self.env.max_move

        cmap = mpl.colors.ListedColormap(['#0000ff', '#3333ff', '#6666ff', '#9999ff', '#ccccff', '#ffffff',
                                          '#ffcccc', '#ff9999', '#ff6666', '#ff3333', '#ff0000'])
        # bounds = [-5.5, -4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        bounds = np.linspace(-self.n / 2, self.n / 2, self.n + 1)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        img = plt.imshow(policy, interpolation='nearest', cmap=cmap, norm=norm, origin='lower')
        plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds, ticks=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
        plt.xlabel('#Cars at second location')
        plt.ylabel('#Cars at first location')
        plt.show()

    def view_values(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        X = np.zeros((self.n, self.n))
        Y = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                X[i, j] = i
                Y[i, j] = j

        values = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                values[i, j] = np.max(self.Q[i, j])

        ax.plot_wireframe(X, Y, values, rstride=self.n-1, cstride=self.n-1)

        plt.show()


class Qlearning(Sarsa):
    def __init__(self, environment):
        Sarsa.__init__(self, environment)

    def learn(self, times):
        self.env.reset()

        reward = 0

        for i in tqdm(range(times)):
            prev_state = np.copy(self.env.state)

            action = self.choose_action()
            _, reward, _ = self.env.one_move(action - self.env.max_move)

            self.Q[prev_state[0], prev_state[1], action] += \
                ALPHA * (reward + DISCOUNT * np.max(self.Q[self.env.state[0], self.env.state[1]])
                         - self.Q[prev_state[0], prev_state[1], action])


if __name__ == '__main__':

    jackCarRental = JacksCarRental()
    sarsa = Sarsa(jackCarRental)
    sarsa.learn(100000)
    sarsa.view_policy()
    sarsa.view_values()
    print('Sarsa done.')

    jackCarRental = JacksCarRental()
    ql = Qlearning(jackCarRental)
    ql.learn(100000)
    ql.view_policy()
    ql.view_values()
    print('Q learning done.')
