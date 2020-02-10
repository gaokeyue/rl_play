import utils


class Agent:
    def __init__(self, game, n_trials=5 * 10 ** 4):
        self.game = game
        self.n_trials = n_trials
        self.competitive_thresh = 0.05

    @staticmethod
    def greedy_pick(action_value_dict, find_one=True, thresh=0):
        if find_one:  # find one maximizer
            return max(action_value_dict, key=action_value_dict.get)
        else:
            best_score = -float('inf')
            best_actions = []
            for action, value in action_value_dict.items():
                flag = utils.compare(value, best_score, thresh)
                if flag is 1:  # a strictly better action is found
                    best_score = value
                    best_actions = [action]
                elif flag is 0:  # an action which ties the best action is found
                    best_actions.append(action)
            return best_actions

    @staticmethod
    def q2pi(q):
        return {state: Agent.greedy_pick(action_value_dict)
                for state, action_value_dict in q.items()}

    @staticmethod
    def compare_dict(d1, d2):
        return [k for k in d1 if d1[k] != d2[k]]

    def policy_run(self, policy, state0=None):
        state = self.game.reset(state0)
        state_ls = []
        reward_ls = []
        is_terminal = False
        while not is_terminal:
            state_ls.append(state)
            action = policy[state]
            state, reward, is_terminal = self.game.one_move(action)
            reward_ls.append(reward)
        return state_ls, reward_ls

    def eval_pi_state_action_one(self, pi, state, action):
        """Given a policy pi, evaluate Q_pi(state, action). Note that action not
        necessarily equals pi[state]
        """
        game = self.game
        result = utils.Averager()
        for _ in range(self.n_trials):
            game.reset(state)
            tmp_a = action
            total_return = 0
            t = 0
            while True:
                tmp_s, reward, is_terminal = game.one_move(tmp_a)
                total_return += reward * game.gamma ** t
                t += 1
                if is_terminal:
                    result.add_new(total_return)
                    break
                else:
                    tmp_a = pi[tmp_s]
        return result.average

    def policy_eval(self, policy, state_action_oi=None):
        """Using Monte-Carlo to evaluate Q_pi(S, A) for (S, A) of interest.
        :param policy (dict) -- only consider deterministic policy, action = policy[state]
        :param state_action_oi (dict) -- (state, action) pair of interest,
        if None, then evaluate every possible state-action pair where state in policy
        """
        if state_action_oi is None:
            state_action_oi = {state: self.game.available_actions(state) for state in policy}
        q = {}
        for state, actions in state_action_oi.items():
            action_value_dict = {}
            for action in actions:
                v = self.eval_pi_state_action_one(policy, state, action)
                action_value_dict[action] = v
                print(F"{state}, {action} --> {v}")
            q[state] = action_value_dict
        return q

    def q_eval(self, q_star):
        sa_oi = {}
        pi_star = self.q2pi(q_star)
        # sift state action of interest
        for state, a_star in pi_star.items():
            a_star = pi_star[state]
            action_value_dict = q_star[state]
            v_star = action_value_dict[a_star]
            sa_oi[state] = [action for action, value in action_value_dict.items()
                            if v_star - value < self.competitive_thresh]
        q_hat = self.policy_eval(pi_star, sa_oi)
        pi_hat = self.q2pi(q_hat)
        # compare pi_star and pi_hat and retrieve states that
        problem_states = [state for state in pi_hat if pi_hat[state] != pi_star[state]]
        return problem_states


if __name__ == '__main__':
    print('haha')
