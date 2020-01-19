from player import Player
from qlearner import Learner
import numpy as np


class Game:
    def __init__(self, num_learning_rounds, learner=None, report_every=100):
        self.p = learner
        self.win = 0
        self.loss = 0
        self.game = 1
        self._num_learning_rounds = num_learning_rounds
        self._report_every = report_every

    def run(self):
        d, player, dealer, winner = self.reset_round()

        state = self.get_starting_state(player, dealer)

        while True:
            player_action = player.get_action(state)
            dealer_action = dealer.get_action(state)
            if player_action == 'hit':
                player.hit(d)
            if dealer_action == 'hit':
                dealer.hit(d)
            if self.determine_if_bust(player):
                winner = 'dealer'
                break
            elif self.determine_if_bust(dealer):
                winner = 'player'
                break
            if player_action == dealer_action and player_action == 'stay':
                break
            state = self.get_state(player, player_action, dealer)
            player.update(state, 0, self.game)

        if winner is None:
            winner = self.determine_winner(player, dealer)

        if winner == 'player':
            self.win += 1
            player.update(self.get_ending_state(player, player_action, dealer), 1, self.game)
        else:
            self.loss += 1
            player.update(self.get_ending_state(player, player_action, dealer), -1, self.game)

        self.game += 1

        # self.report()

        if self.game == self._num_learning_rounds:
            print("Turning off learning!")
            self.p._learning = False
            self.win = 0
            self.loss = 0

    def report(self):
        if self.game % self._num_learning_rounds == 0:
            print(str(self.game) + " : " + str(self.win / (self.win + self.loss)))
        elif self.game % self._report_every == 0:
            print(str(self.win / (self.win + self.loss)))

    def get_state(self, player1, player_action, player2):
        return (player1.get_hand_value(), player2.get_original_showing_value())

    def get_starting_state(self, player1, player2):
        return (player1.get_hand_value(), player2.get_showing_value())

    def get_ending_state(self, player1, player_action, player2):
        return (player1.get_hand_value(), player2.get_hand_value())

    def determine_winner(self, player1, player2):
        if player1.get_hand_value() == 21 or (player1.get_hand_value() > player2.get_hand_value() and player1.get_hand_value() <= 21):
            return 'player'
        else:
            return 'dealer'

    def determine_if_bust(self, player):
        if player.get_hand_value() > 21:
            return True
        else:
            return False

    def reset_round(self):
        d = Deck()
        if self.p is None:
            self.p = Learner()
        else:
            self.p.reset_hand()

        p = self.p
        dealer = Player()

        winner = None
        p.hit(d)
        dealer.hit(d)
        p.hit(d)
        dealer.hit(d)

        return d, p, dealer, winner


class Deck:
    def __init__(self):
        self._cards = []
        self.shuffle()

    def shuffle(self):
        cards = (np.arange(0, 10) + 1)
        cards = np.repeat(cards, 4*3)  # 4 suits x 3 decks
        np.random.shuffle(cards)
        self._cards = cards.tolist()

    def draw(self):
        return self._cards.pop()
