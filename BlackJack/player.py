import random

def draw_cards(n_cards=1):
    cards = range(1, 11)
    pmf = [1] * 9 + [4]
    return random.choices(cards, pmf, k=n_cards)

class Player:
    def __init__(self, value=0):
        self.value = value
        self._hand = []
        self._original_showing_value = 0

    @property
    def goes_bust(self):
        return self.value > 21

    def get_hand(self):
        return self._hand

    def get_action(self):
        raise NotImplementedError

    def get_hand_value(self):
        return sum(self._hand)

    def get_showing_value(self):
        showing = self._hand[1]
        self._original_showing_value = showing
        return showing

    def get_original_showing_value(self):
        return self._original_showing_value

    def hit(self, n_cards=1):
        card = draw_cards(n_cards)
        self.value += sum(card)

    @staticmethod
    def stay():
        return True

    def reset_hand(self, value=0):
        self.value = value

    def update(self):
        pass


class Dealer(Player):
    def get_action(self):
        return 'hit' if self.get_hand_value() < 17 else 'stay'
