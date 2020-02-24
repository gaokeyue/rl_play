import random
import numpy as np

from game import Game


class JacksCarRental(Game):

    gamma = 0.9

    def __init__(self):
        self.rent_price = 10
        self.move_cost = 2
        self.parking_cap = 20
        self.max_move = 5
        self.lambda_rent_loc1 = 3
        self.lambda_return_loc1 = 3
        self.lambda_rent_loc2 = 4
        self.lambda_return_loc2 = 2
        self._state = (random.randint(0, self.parking_cap), random.randint(0, self.parking_cap))
        self.state_space = tuple((s1, s2) for s1 in range(self.parking_cap+1) for s2 in range(self.parking_cap+1))

    @property
    def state(self):
        return self._state

    def reset(self, *states):
        if len(states) > 0:
            self._state = random.choice(states)
        else:
            self._state = (random.randint(0, self.parking_cap), random.randint(0, self.parking_cap))
        return self._state

    def one_move(self, action):

        self._state = (self._state[0] - action, self._state[0] + action)
        reward = - abs(action) * self.move_cost

        cars_rented_loc1 = np.random.poisson(self.lambda_rent_loc1, 1)[0]
        cars_rented_loc1 = min(cars_rented_loc1, self._state[0])
        cars_rented_loc2 = np.random.poisson(self.lambda_rent_loc2, 1)[0]
        cars_rented_loc2 = min(cars_rented_loc2, self._state[1])
        self._state = (self._state[0] - cars_rented_loc1, self._state[1] - cars_rented_loc2)
        reward += (cars_rented_loc1 + cars_rented_loc2) * self.rent_price

        cars_returned_loc1 = np.random.poisson(self.lambda_return_loc1, 1)[0]
        cars_left_loc1 = self._state[0] + cars_returned_loc1
        cars_returned_loc2 = np.random.poisson(self.lambda_return_loc2, 1)[0]
        cars_left_loc2 = self._state[1] + cars_returned_loc2
        self._state = (min(self.parking_cap, cars_left_loc1), min(self.parking_cap, cars_left_loc2))

        return self.state, reward, False

    def available_actions(self, state=None):
        if state is None:
            state = self._state
        max_loc1_can_get = min(self.max_move, self.parking_cap - state[0])
        max_loc2_can_get = min(self.max_move, self.parking_cap - state[1])
        max_loc1_can_move = min(self.max_move, state[0])
        max_loc2_can_move = min(self.max_move, state[1])
        max_loc1to2 = min(max_loc1_can_move, max_loc2_can_get)
        max_loc2to1 = -min(max_loc2_can_move, max_loc1_can_get)
        return range(max_loc2to1, max_loc1to2 + 1)


if __name__ == '__main__':
    jackCarRental = JacksCarRental()
    state = jackCarRental.state
    # state_ls = []
    # action_ls = []
    # reward_ls = []
    # for _ in range(10000):
    #     state_ls.append(state)
    #     action = random.choice(jackCarRental.available_actions(state))
    #     action_ls.append(action)
    #     state, reward, is_terminal = jackCarRental.one_move(action)
    #     reward_ls.append(reward)
    #     if state == (0, 0):
    #         print('haha', _)
    state_space = jackCarRental.state_space
    state = jackCarRental.reset((1, 1))
