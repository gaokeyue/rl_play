from game import Game
import random


class Gambler(Game):
    def __init__(self, p_h=.4, goal=100, state0=None):
        self.p_h = p_h
        self.goal = goal
        self.state_space = range(self.goal + 1)
        if state0 is None:
            self.reset()
        else:
            self.reset(state0)

    @property
    def state(self):
        return self._state

    def reset(self, *states):
        if len(states) > 0:
            self._state = random.choice(states)
        else:
            self._state = random.choice(self.state_space[1:-1])  # excluding terminal state 0 and goal
        return self._state

    def is_terminal(self, state=None):
        if state is None:
            state = self._state
        return state == 0 or state == self.goal

    def one_move(self, action):
        if random.random() <= self.p_h:  # win
            self._state += action
        else:  # lose
            self._state -= action
        reward = 1 if self._state == self.goal else 0
        return self.state, reward, self.is_terminal()

    def available_actions(self, state=None):
        if state is None:
            state = self._state
        return range(1, min(state, self.goal - state) + 1)

    def copy(self):
        return Gambler(self.p_h, self.goal, self.state)


if __name__ == '__main__':
    gambler = Gambler(goal=15)
    state0 = gambler.state
    action0 = random.choice(gambler.available_actions(state0))
    state1, reward1, is_terminal1 = gambler.one_move(action0)
