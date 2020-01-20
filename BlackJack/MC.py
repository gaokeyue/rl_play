import random
from collections import defaultdict


cards = ['K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2', 'A']
cards_count = {
                'K': 10, 'Q': 10, 'J': 10, 'T': 10, '9': 9, '8': 8,
                '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2, 'A': 11
               }


def dealer(shown_card, seed=None):
    random.seed = seed
    usable_ace = False
    card_two = random.choice(cards)
    count = cards_count[shown_card] + cards_count[card_two]
    if card_two == 'A' or shown_card == 'A':
        usable_ace = True
    if count == 22:
        count = 12
    while count < 17:
        hit_card = random.choice(cards)
        count += cards_count[hit_card]
        if count > 21:
            if not usable_ace:
                return 0
            else:
                usable_ace = False
                count = count - 10
        if hit_card is 'A':
            usable_ace = True
    return count


def exploring_start():
    """
    return a stochastic (state, action) pair which covers all possible state action pairs
    state(shown_card, hands_sum, usable_ace)
    action can be hit or stick
    """
    # usable_ace = False
    # shown_card = random.choice(cards)
    shown_card = 'A'
    hands_sum = random.randint(12, 21)
    # card_one, card_two = random.choices(cards, k=2)
    # hands_sum = cards_count[card_one] + cards_count[card_two]
    # if card_two == 'A' or card_two == 'A':
    #     usable_ace = True
    # if hands_sum == 22:
    #     hands_sum = 12
    # usable_ace = random.choice([True, False])
    usable_ace = False
    action = random.choice(['hit', 'stand'])
    return (shown_card, hands_sum, usable_ace), action


class Blackjack:

    def __init__(self):
        self.gamma = 1
        self.pai = {}
        self.q = defaultdict(int)
        self.r = (defaultdict(list), defaultdict(int))

    def monte_carlo(self):
        for i in range(3 * (10 ** 5)):
            if (i % 10000 == 0) & (i != 0):
                print(f"It's {i} iteration")
                self.policy_improvement()
            reward = -1
            pair_lst = []
            pair = exploring_start()
            (shown_card, hands_sum, usable_ace), action = pair
            if hands_sum == 21:
                action = 'stand'
                reward = 1
                pair = ((shown_card, hands_sum, usable_ace), action)
            # if hands_sum <= 11:
            #     action = 'hit'
            pair_lst.append(pair)
            while action is 'hit':
                new_card = random.choice(cards)
                hands_sum += cards_count[new_card]
                if hands_sum > 21:
                    if (not usable_ace) & (new_card is not 'A'):
                        break
                    else:
                        hands_sum -= 10
                        if (new_card is 'A') & usable_ace:
                            usable_ace = True
                        else:
                            usable_ace = False
                if hands_sum == 21:
                    action = 'stand'
                    continue
                try:
                    action = self.pai[pair[0]]
                except KeyError:
                    action = 'stand' if hands_sum >= 17 else 'hit'
                pair = ((shown_card, hands_sum, usable_ace), action)
                pair_lst.append(pair)
            else:
                dealer_points = dealer(shown_card)
                if hands_sum > dealer_points:
                    reward = 1
                elif hands_sum == dealer_points:
                    reward = 0
            G = 0
            for i, pair in enumerate(reversed(pair_lst)):
                if i == 0:
                    G = self.gamma * G + reward
                    self.r[0][pair].append(G)
                else:
                    G = self.gamma * G
                    self.r[0][pair].append(G)
        else:
            self.policy_improvement()

    def policy_improvement(self):
        for pair, returns in self.r[0].items():
            self.q[pair] = (sum(returns) + self.q[pair] * self.r[1][pair]) / (len(returns) + self.r[1][pair])
            self.r[0][pair] = []
            self.r[1][pair] += len(returns)
            state = pair[0]
            self.pai[state] = 'hit' if self.q[(state, 'hit')] >= self.q[state, 'stand'] else 'stand'
        # else:
        #     print(self.q)
        #     print(self.pai)
        #     print('haha')


if __name__ == '__main__':
    blackjack = Blackjack()
    blackjack.monte_carlo()
    usable_ace = []
    no_usable_ace = []
    print(blackjack.q)
    for state, action in blackjack.pai.items():
        if state[-1]:
            usable_ace.append((*state[:2], action))
        else:
            no_usable_ace.append((*state[:2], action))
    for shown_card, cards_sum, action in no_usable_ace:
            print((cards_sum, action))