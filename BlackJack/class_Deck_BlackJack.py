import random


class DeckBlackJack:
    def __init__(self, deck_number=3):
        self._cards = (list(range(1, 11))*4 + [10]*3*4)*deck_number
        random.shuffle(self._cards)

    def draw(self):
        return self._cards.pop()
