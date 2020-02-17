import abc
from collections import defaultdict


class Game(abc.ABC):
    state_space = None  # developers are highly recommended to list the entire state_space
    gamma = 1  # rate of decay of reward

    @property
    @abc.abstractmethod
    def state(self):
        """The current state is kept in game.state, however this is a read only property.
        setting state, game.state = state is forbidden """

    @abc.abstractmethod
    def reset(self, *states, **kwargs):
        """If states is empty, then reset to the most generic initial state of the given name.
        Else, the initial state is chosen from a set of specific states.
        Must do self.state = initial state and return state"""

    @abc.abstractmethod
    def one_move(self, action):
        """return state, reward, is_terminal.
        The reward are expected to be normalized with reward_max = 1 and reward_min = 0/1.
        """

    @abc.abstractmethod
    def available_actions(self, state=None):
        """If state is None, then return the set of available actions from the current self.space.
        It's possible that action is independent of state.
        """

    def is_terminal(self, state=None):
        actions = self.available_actions(state)
        return len(actions) == 0

    def q_initializer(self):
        """This method is not an abstract method because some games are of the 'after state' type, hence
        should be initialized as value function, e.g. defaultdict(int).
        """
        if self.state_space is None:
            return defaultdict(lambda: defaultdict(int))
        q = {}
        for state in self.state_space:
            actions = self.available_actions(state)
            action_value = {action: 0 for action in actions}
            if len(action_value) == 0:  # actions is empty, i.e. state is terminal
                action_value[None] = 0
            q[state] = action_value
        return q
