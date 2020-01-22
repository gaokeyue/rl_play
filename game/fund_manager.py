from abc import ABC

from game.game import Game
# from collections import defaultdict
import random


class FundManager(Game, ABC):
    def __init__(self):
        self._actions = ['left', 'right'] + list(range(0, 100))
        self.is_terminal = False
        self._state = ('shown_card', 'hands_sum', 'usable_or_not')
        self.cards = ['K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2', 'A']
        self.cards_count = {
            'K': 10, 'Q': 10, 'J': 10, 'T': 10, '9': 9, '8': 8,
            '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2, 'A': 11


    @property
    def state(self):
        return self._state

    def reset(self):
        # usable_ace = False
        # shown_card = random.choice(cards)
        shown_card = 'K'
        hands_sum = random.randint(12, 21)
        # hands_sum = 16
        # card_one, card_two = random.choice
        # s(cards, k=2)
        # hands_sum = cards_count[card_one] + cards_count[card_two]
        # if card_two == 'A' or card_two == 'A':
        #     usable_ace = True
        # if hands_sum == 22:
        #     hands_sum = 12
        # usable_ace = random.choice([True, False])
        usable_ace = False
        state = shown_card, hands_sum, usable_ace
        self._state = state
        return state

    def dealer(self, seed=None):
        random.seed = seed
        usable_ace = False
        card_two = random.choice(self.cards)
        shown_card = self.state[0]
        count = self.cards_count[shown_card] + self.cards_count[card_two]
        if card_two == 'A' or shown_card == 'A':
            usable_ace = True
        if count == 22:
            count = 12
        while count < 17:
            hit_card = random.choice(self.cards)
            count += self.cards_count[hit_card]
            if count > 21:
                if not usable_ace:
                    return 0
                else:
                    usable_ace = False
                    count = count - 10
            if hit_card is 'A':
                usable_ace = True
        return count

    def one_move(self, action):
        if action == 'hit':
            new_card = random.choice(self.cards)
            # print(new_card)
            shown_card, hands_sum, usable_ace = self.state
            hands_sum += self.cards_count[new_card]
            new_state = (shown_card, hands_sum, usable_ace)
            self._state = new_state
            if hands_sum > 21:
                if (not usable_ace) & (new_card is not 'A'):
                    reward = -1
                    self.if_terminal = True
                    new_state = "busted"
                else:
                    if hands_sum == 32:
                        reward = -1
                        self.if_terminal = True
                        new_state = "busted"
                    else:
                        hands_sum -= 10
                        if (new_card is 'A') & usable_ace:
                            usable_ace = True
                        else:
                            usable_ace = False
                        new_state = (shown_card, hands_sum, usable_ace)
                        reward = 0
            else:
                reward = 0
        else:
            self.if_terminal = True
            dealer_pts = self.dealer()
            if self._state[1] < dealer_pts:
                reward = -1
            elif self._state[1] == dealer_pts:
                reward = 0
            else:
                reward = 1
            new_state = (self.state[0], 'stand', self.state[2])
        return new_state, reward, self.is_terminal

    def get_action(self):
        return self._actions



if __name__ == '__main__':
    blackjack = BlackJack()
    blackjack.reset()
    print(blackjack.state)
    print(blackjack.get_action())
    print(blackjack.one_move('stand'))


    # for i in range(0, 10000):
    #     print(i)
    #     Q1 = copy.deepcopy(Q1_origin)
    #     Q2 = copy.deepcopy(Q1_origin)
    #     for p_episode in range(0, 300):
    #         state_now = 'A'
    #         while state_now != 'terminal':
    #             # step1
    #             action = take_action(state_now, Q1, Q2)
    #             # step2
    #             reward = state_action_reward(state_now)
    #             state_next = state_action_state[state_now][action]
    #             # step3
    #             Q1, Q2 = q_choose_update(Q1, Q2, state_now, state_next, action, reward)
    #             state_now = state_next
    #             if action == 'left':
    #                 action_record[i][p_episode] = 1

    # listt = np.sum(action_record, axis=0) / 10000
    # plt.plot(listt)
    # plt.show()

    # action_record = np.zeros((10000, 300))

    # update self.game.Q

    #     def reward(self, state, action):
    #         if state == 'A':
    #             return 0
    #         elif state == 'B':
    #             return random.normalvariate(-0.1, 1)
    #         else:
    #             return 0
    #
    #     # reward_list = [f1()] * action_number
    #     # action_dict = dict(zip(action_list, reward_list))
    #
    #     Q1_origin = {'A': {'left': 0, 'right': 0}, 'B': reward_dict, 'terminal': {'move': 0}}
    #     # Q2_origin = {'A':{'left':0,'right':0}, 'B':reward_dict, 'terminal':{'move':0}}
    #     next_state_dict = dict(zip(action_list, final_list))
    #     state_action_state = {'A': {'left': 'B', 'right': 'terminal'}, 'B': next_state_dict}
    #
    # # step3, update two q dicts
    #
    #
