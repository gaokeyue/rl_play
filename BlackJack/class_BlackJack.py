from BlackJack.player import Player, Dealer
from BlackJack.qlearner import Learner
from BlackJack.class_Deck_BlackJack import DeckBlackJack
import random

class BlackJack:
    ranks = list(range(1, 10)) + [10] * 4
    def __init__(self, player_start=None, dealer_start=None, n_decks=None):
        """parameters to consider: replaceable, do we fix the initial hand for player and dealer,
        player's policy, dealer's policy.
        """
        self.player_start = player_start
        self.dealer_start = dealer_start
        self.n_decks = n_decks  # if n_decks is None, then an infinite deck
        self.stake = 1
        self.reset()

    def replenish(self):
        if self.n_decks is None:
            self.deck = None
        else:
            self.deck = self.ranks * 4 * self.n_decks
            random.shuffle(self.deck)

    def draw(self):
        if self.deck is None:  # draw uniformly from (A,2,3,...,T,J,Q,K)
            return random.choices(self.ranks, weights=[1] * 9 + [4])[0]
        return self.deck.pop()

    def reset(self):
        self.replenish()
        if self.player_start is None:
            player_sum = self.draw()
            player_sum += self.draw()
        else:
            player_sum = self.player_start
        if self.dealer_start is None:
            dealer_sum = self.draw()  # not exact. We may need to differentiate card shown and not shown
        else:
            dealer_sum = self.dealer_start
        return player_sum, dealer_sum

    def one_move(self, state, action):
        """:return is_terminal, reward, new_state"""
        player_sum, dealer_sum = state
        if action == 'hit':
            player_sum += self.deck.draw()
            new_state = (player_sum, dealer_sum)
            if player_sum > 21:
                is_terminal = True
                reward = -self.stake
            else:
                is_terminal = False
                reward = 0
            return is_terminal, reward, new_state
        elif action == 'stick':
            is_terminal = True
            while dealer_sum < 17:
                dealer_sum += self.deck.draw()
                if dealer_sum > 21:
                    return is_terminal, self.stake, (player_sum, dealer_sum)
            new_state = (player_sum, dealer_sum)
            if player_sum > dealer_sum:
                reward = self.stake
            elif player_sum == dealer_sum:
                reward = 0
            else:
                reward = -self.stake
            return is_terminal, reward, new_state
        else:
            raise ValueError("action can only be 'hit' or 'stick'")

    def run(self, player_start=None, dealer_start=None):
        self.reset(player_start, dealer_start)
        # state0 = (player.value, dealer.value)
        # action = player.get_action(state)
        # reward, state1 = game(state, action)
        # if state1 is not terminal, state0 = state1
        player = self.player
        dealer = self.dealer
        winner = None

        # state_action_pair_ls = [state]
        while player.get_action(state) == 'hit':
            player.hit()
            if player.goes_bust:
                winner = 'dealer'
                # player.update(new_state='player bust end', reward=-self.stake, game=self.game_num)
                break
            else:
                state = (player.value, dealer.value)
                # player.update(new_state=state, reward=0, game=self.game_num)
        while dealer.get_action() == 'hit':
            dealer.hit()
            if dealer.goes_bust:
                winner = 'player'
                player.update(new_state='dealer bust end', reward=self.stake, game=self.game_num)
                break
        if winner is None:
            winner = self.determine_winner(player1=player, player2=dealer)
            if winner == 'player':
                player.update(new_state='win end', reward=self.stake, game=self.game_num)
            elif winner == 'dealer':
                self.loss += 1
                player.update(new_state='loss end', reward=-self.stake, game=self.game_num)
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



