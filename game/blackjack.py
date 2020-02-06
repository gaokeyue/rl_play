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
            hands_sum = random.randint(12, 21)
            usable_ace = random.choice([True, False])
        state = shown_card, hands_sum, usable_ace
        self._state = state
        return state

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
        is_terminal = False
        if action == 'hit':
            new_card = random.choice(self.cards)
            # print(new_card)
            shown_card, hands_sum, usable_ace = self._state
            hands_sum += self.cards_count[new_card]
            new_state = (shown_card, hands_sum, usable_ace)
            # self._state = new_state
            if hands_sum > 21:
                if (not usable_ace) & (new_card is not 'A'):
                    reward = -1
                    is_terminal = True
                    new_state = (shown_card, "busted", usable_ace)
                else:
                    if hands_sum == 32:
                        reward = -1
                        is_terminal = True
                        new_state = (shown_card, "busted", usable_ace)
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

    def q_initializer(self):
        q = defaultdict(dict)
        for shown_card in self.cards:
            for hands_sum in (list(range(12, 22)) + ['busted', 'stand']):
                state = (shown_card, hands_sum, True)
                q[state]['hit'] = 0
                q[state]['stand'] = 0
                state = (shown_card, hands_sum, False)
                q[state]['hit'] = 0
                q[state]['stand'] = 0
        return q


if __name__ == '__main__':
    blackjack = BlackJack()
    blackjack.reset()
    print(blackjack._state)
    print(blackjack.available_actions())
    print(blackjack.one_move('hit'))
    print(blackjack.q_initializer())
