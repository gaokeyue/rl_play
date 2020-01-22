import random
from Warlock.class_Game_Warlock import GameWarlock


class QLearning:
    def __init__(self, game, num_learning_rounds):
        self.game = game
        self.Q = {}
        available_actions = self.game.available_actions()
        self.Q.update({self.game.state: {action: 0 for action in available_actions}})
        self.Q_times = {}
        self.last_state = None
        self.last_action = None
        self.learning_rate_origin = 0.1
        self.learning_rate = self.learning_rate_origin
        self.num_learning_rounds = num_learning_rounds
        self.gamma = 0.998
        self.epsilon = 0.1

    def reset(self):
        self.game.reset()
        self.last_state = None
        self.last_action = None

    def get_action(self):
        if random.random() > self.epsilon:
            available_actions = self.game.available_actions()
            if len(available_actions) == 0:
                raise ValueError('No legal action!')
            else:
                if self.game.state in self.Q:
                    best_action = available_actions[0]
                    best_action_value = self.Q[self.game.state][best_action]
                    for action in available_actions[1:]:
                        if self.Q[self.game.state][action] > best_action_value:
                            best_action = action
                            best_action_value = self.Q[self.game.state][action]
                    action = best_action
                else:
                    self.Q.update({self.game.state: {action: 0 for action in available_actions}})
                    action = random.choice(self.game.available_actions())
        else:
            action = random.choice(self.game.available_actions())
        self.last_state = self.game.state
        self.last_action = action
        return action

    def update(self, action):
        self.game.one_move(action=action)
        available_actions = self.game.available_actions()
        if self.game.state in self.Q:
            action = max(self.Q[self.game.state], key=self.Q[self.game.state].get)
            max_q_new = self.Q[self.game.state][action]
        else:
            self.Q.update({self.game.state: {action: 0 for action in available_actions}})
            max_q_new = 0
        self.Q[self.last_state][self.last_action] \
            = (1-self.learning_rate)*self.Q[self.last_state][self.last_action] \
            + self.learning_rate*(self.game.reward+self.gamma * max_q_new)

    def show_policy(self):
        self.reset()
        while not self.game.is_terminal:
            available_actions = self.game.available_actions()
            best_action = available_actions[0]
            best_action_value = self.Q[self.game.state][best_action]
            for action in available_actions[1:]:
                if self.Q[self.game.state][action] > best_action_value:
                    best_action = action
                    best_action_value = self.Q[self.game.state][action]
            action = best_action
            buff = ''
            if self.game.warlock.buff['dark_soul_buff']['remaining_duration'] > 0:
                buff += 'dark_soul '
            if self.game.warlock.buff['deflagration_buff']['remaining_duration'] > 0:
                buff += 'deflagration '
            print('skill: ', action, 'ashes: ', round(self.game.warlock.ashes/10, 1))
            print('buff: ', buff, 'CD:', self.game.warlock.cd)
            self.game.one_move(action=action)
            print('boss_HP: ', self.game.boss_current_hp)
            print('')

    def run(self):
        for episodes in range(self.num_learning_rounds):
            self.reset()
            self.learning_rate = self.learning_rate_origin * (0.99 ** (episodes // 100))
            while not self.game.is_terminal and self.game.battle_duration < 99999:
                action = self.get_action()
                self.update(action=action)
        self.show_policy()
        print('battle_duration: ', self.game.battle_duration)


if __name__ == '__main__':

    Hasky = QLearning(game=GameWarlock(boss_hp=100000), num_learning_rounds=100000)
    Hasky.run()
