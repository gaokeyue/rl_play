from collections import defaultdict
import numpy as np
import pandas as pd
import copy
from game import blackjack


def e_greedy_policy_creation(Qstate, epsilon, nA):
    """
    Q: Our Q table.
      Q[state] = numpy.array
      Q[state][action] = float.
    epsilon: small value that controls exploration.
    nA: the number of actions available in this environment

    return: an epsilon-greedy policy specific to the state.
    """
    policy = np.ones(nA) * epsilon / nA
    policy[np.argmax(Qstate)] = 1 - epsilon + (epsilon / nA)
    return policy


def choose_action(policy, env):
    return np.random.choice(env.available_actions(), p=policy)


def generate_episode_from_policy_and_state(env, policy, state):
    episode = []
    # ?? This shouldn't reset here since we want to go from the given state
    # I should probably break into the blackjack env to see how they do env.step(action).
    # print(state)
    state = env.reset(state)
    while True:
        action = choose_action(policy, env)
        next_state, reward, done = env.one_move(action)  # ??
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode


class Node:
    def __init__(self, state, action, value):
        self.children = {}
        self.state = state
        self.action = action
        self.value = value
        self.parent = -1

    def isBarren(self):
        return True if len(self.children) == 0 else False

    def addChild(self, state, action, value):
        self.children[state] = Node(state, action, value)
        self.children[state].parent = self.state

    def __str__(self):
        s = "children: {}".format(self.children)
        return s


class MCTS:
    """ Implmentation of UCB MCTS in Blackjack """

    def __init__(self):
        self.root = defaultdict(lambda: Node((0, 0, False), 0, 0))
        self.initial_state = env.reset()
        self.root[self.initial_state] = Node(self.initial_state, 0, 0)
        self.rolloutPolicy = None
        self.Q = defaultdict(lambda: np.ones(2) / 2)
        self.UCB_C = 2
        self.timeStep = 0
        self.kAction = np.zeros(2)
        self.epsilon = 1 / (self.timeStep + 1)
        self.gamma = 1
        self.alpha = 1e-3

    def UCB_action_choice(self, state):
        # avoid 0/0 with time + 1 and kAction + delta
        UCB_estimation = self.Q[state] + \
                         self.UCB_C * (np.sqrt(np.log(self.timeStep + 1) / (self.kAction + 1e-5)))

        # find the best action
        action_max = np.argmax(UCB_estimation)

        # if multiple actions are the best, chose randomly
        other_choices = []
        for action, q in enumerate(UCB_estimation):
            # this will contain action_max
            if q == UCB_estimation[action_max]:
                other_choices.append(action)

        a = np.random.choice(other_choices)

        self.kAction[a] += 1
        self.timeStep += 1

        return a

    def findNodeLocation(self, root, nodeId):
        n = None
        if root.state == nodeId:
            return root
        for k in root.children:
            ans = self.findNodeLocation(root.children[k], nodeId)
            if ans is not None:
                n = ans
        return n

    def addChild(self, action, parent, chain):
        new_state, reward = self.get_new_state(parent.state, action)
        parent.addChild(new_state, action, reward)
        chain.append((parent.state, action, reward))

    def get_new_state(self, state, action):
        # will need a slightly different version of step. same as above.
        # new_state, reward, done, info = self.env.act(action)
        # print(state, action)
        state = env.reset(state)
        next_state, reward, done = env.one_move(action)

        return next_state, reward

    def deconstruct(self, root, leaves):
        """ Starting at the root node, a tree policy based on the action values attached 
        to the edges of the tree traverses the tree to select a leaf node """
        leaves[root.state] = list(root.children)

        for k in root.children:
            self.deconstruct(root.children[k], leaves)

    def follow_tree_policy(self, root, leaves, chain):
        """ Apply UCB at each node in the tree to move to the next node """
        if root.state in leaves:
            return root.state

        action = self.UCB_action_choice(root.state)
        chain.append((root.state, action, root.value))

        leaf = root.state

        for k in root.children:
            if action == root.children[k].action:
                leaf = self.follow_tree_policy(root.children[k], leaves, chain)
        return leaf

    def selection(self, root, chain):
        """ Starting at the root node, a tree policy based on the action values attached 
        to the edges of the tree traverses the tree to select a leaf node """
        # find leaf/unexplored states

        l = {}
        self.deconstruct(root, l)
        leaves = []
        for k in l:
            # if it has zero or one children, this node is a possible terminal node.
            if len(l[k]) <= 1:
                leaves.append(k)

        chain.append((self.initial_state, 0, 0))
        leaf = self.follow_tree_policy(root, leaves, chain)
        # print(leaf)
        return leaf

    def expansion(self, stateId, chain):
        """ For the chosen state, choose an action to expand the tree from """
        parentNode = self.findNodeLocation(self.root[self.initial_state], stateId)
        if parentNode is None:
            raise Exception("node: {} is not in tree".format(stateId))
        # print(parentNode.state)
        a = self.UCB_action_choice(parentNode.state)
        # print(a)

        self.addChild(a, parentNode, chain)
        return parentNode

    def simulation(self, numSims, leaf):
        """ rollout algorithm """
        episodes = []
        for i in range(numSims):
            episodes.append(generate_episode_from_policy_and_state(env, self.rolloutPolicy, leaf))
        return episodes

    def backup(self, chain):
        """ backup the value we calculated from the rollout """

        seen = []

        for t, (state, action, reward) in enumerate(chain):
            action_idx = 1 if action == 'hit' else 0

            if state not in seen:
                seen.append(state)
                G = 0

                for fi, (fstate, faction, freward) in enumerate(chain[t:]):
                    G += (self.gamma ** fi) * freward

                self.Q[state][action_idx] += self.alpha * (G - self.Q[state][action_idx])
        return

    def MC(self, startState):
        """ do all the steps for MCTS """
        chain = []

        # select a leaf node
        state = self.selection(m.root[self.initial_state], chain)

        # expand that leaf node
        leaf = self.expansion(state, chain)

        # generate a rolloutPolicy for said leaf node
        # However, why should that be used at each step of the rollout?
        # I don't think it should be. 
        self.rolloutPolicy = e_greedy_policy_creation(self.Q[leaf.state], self.epsilon, 2)

        # simulate a bunch of MC episodes to get value of said state
        episode = self.simulation(100, leaf.state)

        # remove placeholder starting state from chain
        chain.remove(chain[0])

        for e in episode:
            c = copy.deepcopy(chain)
            c.extend(e)
            # print(c)
            self.backup(c)
        return


if __name__ == '__main__':
    env = blackjack.BlackJack()
    m = MCTS()
    for _ in range(100):
        m.MC(m.initial_state)
    q_table = pd.DataFrame(m.Q).T
    print(q_table)
