"""
softmax-Q actor-criticモデル
"""

import numpy as np
import random


# softmaxアルゴリズム
class softmax(object):
    def softmax(self, a):
        c = np.max(a)
        exp_a = np.exp(a - c)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a

        return y

    def serect_action(self, value, current_state):
        value_p = self.softmax(value[current_state])[0]
        action = np.random.choice(len(value_p), 1, value_p.tolist())[0]
        return action


# TD学習アルゴリズム
class Q_learning(object):

    def __init__(self, learning_rate, discount_rate, num_state, num_action):
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.num_state = num_state
        self.num_action = num_action
        self.Q = np.zeros((num_state, num_action))
        self.TD = None

    def update_params(
            self, current_state, next_state,
            current_action, reward, next_action=None):
        maxQ = max(self.Q[next_state])
        self.TD_error = (reward
                         + self.discount_rate
                         * maxQ
                         - self.Q[current_state, current_action])
        self.Q[current_state, current_action] += self.learning_rate * self.TD_error

    def init_params(self):
        self.Q = np.zeros((self.num_state, self.num_action))

    def get_Q(self):
        return self.Q


class critic(Q_learning):
    def __init__(self, learning_rate=0.1, discount_rate=0.9, num_state=None, num_action=None):
        super().__init__(self, learning_rate, discount_rate, num_state, num_action)

    def get_TDerror(self):
        return self.TD_error


class actor():
    def __init__(self, learning_rate=0.1, num_state=None, num_action=None):
        self.num_state = num_state
        self.num_action = num_action
        self.weight = np.zeros(num_state, num_action)
        self.learning_rate = learning_rate
    
    def serect_action(self, current_state):
        value_p = softmax.softmax(self.weight[current_state])[0]
        action = np.random.choice(len(value_p), 1, value_p.tolist())[0]
        return action

    def update_params(self, TD_error, current_state, current_action):
        self.weight[current_state, current_action] += self.learning_rate * TD_error

    def init_params(self):
        self.weight = np.zeros(self.num_state, self.num_action)


class actor_critic(object):
    def __init__(self, learning_rate, discount_rate, num_state, num_action):
        self.critic = critic(num_state=num_state, num_action=num_action)
        self.actor = actor(num_state=num_state, num_action=num_action)
    
    # 行動選択
    def serect_action(self, current_state):
        return self.actor.serect_action(current_state)

    # パラメータ更新
    def update(self, current_state, current_action, reward, next_state):
        self.critic.update_params(current_state, current_action, reward, next_state)
        self.actor.update_params(self.critic.get_TDerror, current_state, current_action)

    # 所持パラメータの初期化
    def init_params(self):
        self.actor.init_params()
        self.critic.init_params()
