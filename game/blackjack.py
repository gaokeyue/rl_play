from game import Game
from collections import defaultdict
import random


class BlackJack(Game):

    def __init__(self):
        self._actions = ['hit', 'stand']
        self._state = ('shown_card', 'hands_sum', 'usable_or_not')
        self.cards = ['K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2', 'A']
        self.cards_count = {
            'K': 10, 'Q': 10, 'J': 10, 'T': 10, '9': 9, '8': 8,
            '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2, 'A': 11
        }
        self.state_space = [(shown_card, hands_sum, usable_ace)
                            for shown_card in self.cards
                            for hands_sum in range(11, 22)
                            for usable_ace in (True, False)]

    @property
    def state(self):
        return self._state

    def reset(self, *states):
        """
        state: (shown_card, hands_sum, usable_ace)
        """
        # usable_ace = False
        if len(states) > 0:
            shown_card, hands_sum, usable_ace = random.choice(states)
        else:
            shown_card = random.choice(self.cards)
            hands_sum = random.randint(11, 21)
            usable_ace = random.choice([True, False])
            # usable_ace = False
        state = shown_card, hands_sum, usable_ace
        self._state = state
        return state

    def hit(self, hands_sum, hit_card, usable_ace):
        hands_sum = hands_sum + self.cards_count[hit_card]
        if (hit_card != 'A') and not usable_ace:
            pass
        elif usable_ace and (hit_card == 'A'):
            if hands_sum > 21:
                hands_sum = hands_sum - 10
                usable_ace = True
            if hands_sum > 21:
                hands_sum = hands_sum - 10
                usable_ace = False
        else:
            usable_ace = True
            if hands_sum > 21:
                hands_sum = hands_sum - 10
                usable_ace = False
        return hands_sum, usable_ace

    def dealer(self, seed=None):
        random.seed = seed
        usable_ace = False
        card_two = random.choice(self.cards)
        shown_card = self._state[0]
        count = self.cards_count[shown_card] + self.cards_count[card_two]
        if card_two == 'A' or shown_card == 'A':
            usable_ace = True
        if count == 22:
            count = 12
        while count < 17:
            hit_card = random.choice(self.cards)
            count, usable_ace = self.hit(count, hit_card, usable_ace)
            if count > 21:
                return 0
        return count

    def one_move(self, action):
        is_terminal = False
        if action == 'hit':
            new_card = random.choice(self.cards)
            # print(new_card)
            shown_card, hands_sum, usable_ace = self._state
            hands_sum, usable_ace = self.hit(hands_sum, new_card, usable_ace)
            new_state = (shown_card, hands_sum, usable_ace)
            # self._state = new_state
            if hands_sum > 21:
                reward = -1
                is_terminal = True
                new_state = (shown_card, 'busted', usable_ace)
            else:
                reward = 0
        else:
            is_terminal = True
            dealer_pts = self.dealer()
            if self._state[1] < dealer_pts:
                reward = -1
            elif self._state[1] == dealer_pts:
                reward = 0
            else:
                reward = 1
            new_state = (self._state[0], 'stand', self._state[2])
        self._state = new_state
        return new_state, reward, is_terminal

    def available_actions(self, state=None):
        return self._actions

    def policy_initializer(self):
        """
        :return: the initial policy should be a deterministic policy
        """
        return {state: 'stand' if state[1] >= 20 else 'hit' for state in self.state_space}


if __name__ == '__main__':
    blackjack = BlackJack()
    blackjack.reset()
    print(blackjack._state)
    print(blackjack.available_actions())
    print(blackjack.one_move('hit'))
    print(blackjack.q_initializer())
