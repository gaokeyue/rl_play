from game import Game
import random


class Gambler(Game):
    def __init__(self, p_h=.4, goal=100):
        self.p_h = p_h
        self.goal = goal
        self.state_space = range(self.goal + 1)
        self.reset()

    def reset(self, *states):
        if len(states) > 0:
            self.state = random.choice(states)
        else:
            self.state = random.choice(self.state_space[1:-1])  # excluding terminal state 0 and goal
        return self.state

    def is_terminal(self, state=None):
        if state is None:
            state = self.state
        return state == 0 or state == self.goal

    def one_move(self, action):
        if random.random() <= self.p_h:  # win
            self.state += action
        else:  # lose
            self.state -= action
        reward = 1 if self.state == self.goal else 0
        return self.state, reward, self.is_terminal()

    def available_actions(self, state=None):
        if state is None:
            state = self.state
        return range(1, min(state, self.goal - state) + 1)

if __name__ == '__main__':
    gambler = Gambler()
