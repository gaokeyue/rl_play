import time
import math
import random
from copy import deepcopy

from game.true_blackjack import BlackJack


def random_policy(state):
    while not state.is_terminal:
        try:
            action = random.choice(state.get_possible_actions())
        except IndexError:
            raise Exception("Non-terminal state has no possible actions: " + str(state))
        state = state.take_action(action)
    return state.get_reward()


class StateInterface:
    def __init__(self, env):
        self.env = env
        self.state = env.state
        self.reward = 1
        self.is_terminal = False

    def get_possible_actions(self, state=None):
        return self.env.available_actions(state)

    def take_action(self, action):
        env = deepcopy(self.env)
        self.state, self.reward, self.is_terminal = env.one_move(action)
        return deepcopy(self)

    def get_reward(self):
        return self.reward

    def get_terminal(self):
        return self.is_terminal

    def __repr__(self):
        return f"{self.state}, {self.reward}, {self.is_terminal}"


class TreeNode:
    def __init__(self, state, parent):
        self.state = state
        self.isTerminal = state.is_terminal
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        self.children = {}

    def __repr__(self):
        return f"Node({self.state})"


class MCTS:
    def __init__(self, time_limit=None, iteration_limit=None, exploration_constant=1 / math.sqrt(2),
                 rollout_policy=random_policy):
        if time_limit is not None:
            if iteration_limit is not None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.timeLimit = time_limit
            self.limitType = 'time'
        else:
            if iteration_limit is None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if iteration_limit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.searchLimit = iteration_limit
            self.limitType = 'iterations'
        self.explorationConstant = exploration_constant
        self.rollout = rollout_policy
        self.root = None

    def __repr__(self):
        return f"Tree(root={self.root})"

    def search(self, initial_state):
        self.root = TreeNode(initial_state, None)

        if self.limitType == 'time':
            time_limit = time.time() + self.timeLimit / 1000
            while time.time() < time_limit:
                self.execute_round()
        else:
            for i in range(self.searchLimit):
                self.execute_round()

        best_child = self.get_best_child(self.root, 0)
        return self.get_action(self.root, best_child)

    def execute_round(self):
        node = self.select_node(self.root)
        reward = self.rollout(node.state)
        self.backpropogate(node, reward)

    def select_node(self, node):
        while not node.isTerminal:
            if node.isFullyExpanded:
                node = self.get_best_child(node, self.explorationConstant)
            else:
                return self.expand(node)
        return node

    @staticmethod
    def expand(node):
        actions = node.state.get_possible_actions()
        for action in actions:
            if action not in node.children:
                new_node = TreeNode(node.state.take_action(action), node)
                node.children[action] = new_node
                if len(actions) == len(node.children):
                    node.isFullyExpanded = True
                return new_node

        raise Exception("Should never reach here")

    @staticmethod
    def backpropogate(node, reward):
        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent

    @staticmethod
    def get_best_child(node, exploration_value):
        best_value = float("-inf")
        best_nodes = []
        for child in node.children.values():
            node_value = child.totalReward / child.numVisits + exploration_value * math.sqrt(
                2 * math.log(node.numVisits) / child.numVisits)
            if node_value > best_value:
                best_value = node_value
                best_nodes = [child]
            elif node_value == best_value:
                best_nodes.append(child)
        return random.choice(best_nodes)

    @staticmethod
    def get_action(root, best_child):
        for action, node in root.children.items():
            if node is best_child:
                return action


if __name__ == '__main__':
    game = BlackJack()
    state_of_interest = ('4', 12, False)
    game.reset(state_of_interest)
    print(game.state)
    init_state = StateInterface(game)
    mc_tree = MCTS(iteration_limit=1000000)
    best_action = mc_tree.search(initial_state=init_state)
    print(best_action)
