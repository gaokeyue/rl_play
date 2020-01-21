from game import Game
from collections import defaultdict
import random


class BlackJack(Game):

    def __init__(self):
        self.q = defaultdict(dict)
        self._actions = ['hit', 'stand']
        self.is_terminal = False
        self._state = ('shown_card', 'hands_sum', 'usable_or_not')
        self.cards = ['K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2', 'A']
        self.cards_count = {
            'K': 10, 'Q': 10, 'J': 10, 'T': 10, '9': 9, '8': 8,
            '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2, 'A': 11
        }

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
            new_state = ('stand', self.state[1], self.state[2])
        return new_state, reward, self.is_terminal

    def get_action(self):
        return self._actions


blackjack = BlackJack()

if __name__ == '__main__':
    blackjack.reset()
    print(blackjack.state)
    print(blackjack.get_action())
    print(blackjack.one_move('hit'))