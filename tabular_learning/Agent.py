class Agent:
    def __init__(self, env):
        self.env = env

    def value_eval(self, value):
        pass

    def policy_eval_on(self, policy):
        """Using Monte-Carlo to evaluate a given policy"""
        pass

    def policy_eval_off(self, policy):
        pass

    def mc_control_on(self):
        pass

    def mc_control_off(self):
        pass

    def sarsa(self):
        pass

    def q_learning(self):
        pass

    def expected_sarsa(self):
        pass

    def double_q(self):
        pass