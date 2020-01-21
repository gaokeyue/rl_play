import abc

class Game(abc.ABC):

    @property
    @abc.abstractmethod
    def state(self):
        """return the current state"""

    @abc.abstractmethod
    def reset(self):
        """self.state = initial state and return state"""

    @abc.abstractmethod
    def one_move(self, action):
        """return state, reward, is_terminal, debug_info"""

    @abc.abstractmethod
    def get_action(self):
        """return the action space from self.space. It's possible that action is independent of state"""

