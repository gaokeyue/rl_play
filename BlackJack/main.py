from game import Game
from qlearner import Learner
import matplotlib.pyplot as plt


def main():
    num_learning_rounds = 100000
    game = Game(num_learning_rounds, Learner(num_learning_rounds))  # Q learner
    number_of_test_rounds = 1000
    for k in range(0, num_learning_rounds):
        game.run()

    df = game.p.get_optimal_strategy()
    df = df.sort_index()
    return game, df


if __name__ == "__main__":
    g, p = main()
    print(p.groupby(level=1).get_group(10)[10:])
    # p.to_csv('optimal_policy.csv')
    # plt.plot(g.p._hit)
    # plt.plot(g.p._stay)
    # plt.show()
