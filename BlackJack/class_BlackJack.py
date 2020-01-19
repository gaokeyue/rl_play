from BlackJack.player import Player
from BlackJack.qlearner import Learner
from BlackJack.class_Deck_BlackJack import DeckBlackJack


class BlackJack:
    def __init__(self, num_learning_rounds, learner=None, report_every=100):
        self.player = learner
        self.win = 0
        self.loss = 0
        self.game_num = 1
        self._num_learning_rounds = num_learning_rounds
        self._report_every = report_every

    def reset_round(self):
        deck = DeckBlackJack(deck_number=8)
        if self.player is None:
            self.player = Learner(num_learning_rounds=self._num_learning_rounds)
        else:
            self.player.reset_hand()
        player = self.player
        dealer = Player()
        winner = None
        player.hit(deck=deck)
        dealer.hit(deck=deck)
        player.hit(deck=deck)
        dealer.hit(deck=deck)
        return deck, player, dealer, winner

    def run(self):
        deck, player, dealer, winner = self.reset_round()
        state = self.get_starting_state(player1=player, player2=dealer)
        while player.get_action(state=state) == 'hit':
            player.hit(deck=deck)
            if self.determine_if_bust(player=player):
                winner = 'dealer'
                self.loss += 1
                player.update(new_state='player bust end', reward=-1, game=self.game_num)
                break
            else:
                state = self.get_state(player1=player, player2=dealer)
                player.update(new_state=state, reward=0, game=self.game_num)
        while dealer.get_action() == 'hit':
            dealer.hit(deck=deck)
            if self.determine_if_bust(player=dealer):
                winner = 'player'
                self.win += 1
                player.update(new_state='dealer bust end', reward=1, game=self.game_num)
                break
        if winner is None:
            winner = self.determine_winner(player1=player, player2=dealer)
            if winner == 'player':
                self.win += 1
                player.update(new_state='win end', reward=1, game=self.game_num)
            elif winner == 'dealer':
                self.loss += 1
                player.update(new_state='loss end', reward=-1, game=self.game_num)
            else:
                player.update(new_state='draw end', reward=0, game=self.game_num)

        self.game_num += 1
        # self.report()
        if self.game_num == self._num_learning_rounds:
            print("Turning off learning!")
            self.player._learning = False

    def report(self):
        if self.game_num % self._num_learning_rounds == 0:
            print(str(self.game_num) + " : " + str(self.win / (self.win + self.loss)))
        elif self.game_num % self._report_every == 0:
            print(str(self.win / (self.win + self.loss)))

    @staticmethod
    def get_state(player1, player2):
        return player1.get_hand_value(), player2.get_original_showing_value()

    @staticmethod
    def get_starting_state(player1, player2):
        return player1.get_hand_value(), player2.get_showing_value()

    @staticmethod
    def determine_winner(player1, player2):
        if player1.get_hand_value() == 21 \
                or 21 >= player1.get_hand_value() > player2.get_hand_value():
            return 'player'
        elif player1.get_hand_value() == player2.get_hand_value():
            return None
        else:
            return 'dealer'

    @staticmethod
    def determine_if_bust(player):
        if player.get_hand_value() > 21:
            return True
        else:
            return False



