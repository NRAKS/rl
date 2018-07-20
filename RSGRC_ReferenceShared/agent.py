"""
Python3
内容:
    行動価値を使用した学習方法をまとめたクラス
    ε-greedyなどの行動決定アルゴリズム
"""

import random
import numpy as np
import math
import sys


# TD学習アルゴリズム
class Q_learning(object):

    def __init__(self, learning_rate, discount_rate, num_state, num_action):
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.num_state = num_state
        self.num_action = num_action
        self.Q = np.zeros((num_state, num_action))

    def update_Q(
            self, current_state, next_state,
            current_action, reward, next_action=None):
        maxQ = max(self.Q[next_state])
        TD = (reward
              + self.discount_rate
              * maxQ
              - self.Q[current_state, current_action])
        self.Q[current_state, current_action] += self.learning_rate * TD

    def init_params(self):
        self.Q = np.zeros((self.num_state, self.num_action))

    def get_Q(self):
        return self.Q


# sarsa学習
class Sarsa(Q_learning):
    def __init__(self, learning_rate, discount_rate, num_state, num_action):
        super.__init__(learning_rate, discount_rate, num_state, num_action)

    def update_Q(
            self, current_state, next_state,
            current_action, reward, next_action):
        next_Q = self.Q[next_state, next_action]
        TD = (reward
              + self.discount_rate
              * next_Q
              - self.Q[current_state, current_action])
        self.Q[current_state, current_action] += self.learning_rate * TD


# RS(risk-sensitive satisficing)モデル
class RS(object):
    def __init__(
            self, learning_rate, discount_rate, reference,
            tau_alpha, tau_gamma, num_state, num_action, policy):
        self.reference_init = reference
        self.reference = np.full(num_state, reference)
        self.tau_alpha = tau_alpha
        self.tau_gamma = tau_gamma
        self.num_action = num_action
        self.num_state = num_state
        self.tau = np.zeros((num_state, num_action))
        self.tau_current = np.zeros((num_state, num_action))
        self.tau_post = np.zeros((num_state, num_action))

        # 方策によってQ学習とsarsaを切り替える
        if policy == "Q_learning":
            self.policy = Q_learning(learning_rate, discount_rate,
                                     num_state, num_action)
        elif policy == "sarsa":
            self.policy = Sarsa(learning_rate, discount_rate,
                                num_state, num_action)
        else:
            sys.exit("Error")

    def get_serect_action(self, current_state): 
        Q = self.policy.get_Q()
        rs = (self.tau[current_state]
              * (Q[current_state]
              - self.reference[current_state]))

        idx = np.where(rs == max(rs))
        serect_action = random.choice(idx[0])
        return serect_action

    def update(
            self, current_state, next_state,
            current_action, reward, next_action=None):
        self.policy.update_Q(current_state, next_state,
                             current_action, reward, next_action)
        # τ値更新準備
        max_next_state_Q = max(self.policy.get_Q()[next_state])
        idx = np.where(self.policy.get_Q()[next_state] == max_next_state_Q)
        action_up = random.choice(idx[0])
        # τ値更新
        self.tau_current[current_state, current_action] += 1
        self.tau_post[current_state, current_action] += (self.tau_alpha
                                                         * (self.tau_gamma
                                                            * self.tau[next_state, action_up]
                                                            - self.tau_post[current_state, current_action]))

        self.tau[current_state, current_action] = (self.tau_current[current_state, current_action]
                                                   + self.tau_post[current_state, current_action])

    def init_params(self):
        self.policy.init_params()
        self.reference = np.full(self.num_state, self.reference_init)
        self.tau = np.zeros((self.num_state, self.num_action))
        self.tau_current = np.zeros((self.num_state, self.num_action))
        self.tau_post = np.zeros((self.num_state, self.num_action))


class GRC(RS):
    def __init__(
            self, learning_rate, discount_rate, reference, zeta,
            tau_alpha, tau_gamma, num_state, num_action, policy):
        super().__init__(learning_rate, discount_rate, reference,
                         tau_alpha, tau_gamma, num_state, num_action, policy)
        self.RG = reference
        self.zeta = zeta
        self.EG = 0
        self.NG = 0
        self.GRC_gamma = discount_rate
        self.NG_R = 0

    def get_serect_action(self, current_state):
        DG = min([self.EG - self.RG, 0])
        Q = self.policy.get_Q()
        max_Q = max(Q[current_state])
        reference = max_Q - self.zeta * DG
        rs = {}
        rs[current_state] = (self.tau[current_state]
                             * (Q[current_state]
                             - reference))
        idx = np.where(rs[current_state] == max(rs[current_state]))

        serect_action = random.choice(idx[0])

        return serect_action

    # culculation EG
    def update_GRC_params(self, sum_reward):
        self.Etmp = sum_reward
        self.EG = ((self.Etmp
                   + self.GRC_gamma
                   * (self.NG * self.EG))
                   / (1 + self.GRC_gamma * self.NG))
        self.NG = 1 + self.GRC_gamma * self.NG

    def init_params(self):
        super().init_params()
        self.EG = 0
        self.NG = 0

    def update_RG(self, RG_up):
        self.RG = RG_up

    # culculation RG
    def update_GRC_reference(self, R_up):
        self.Rtmp = R_up

        self.RG = ((self.Rtmp
                   + self.GRC_gamma
                   * (self.NG_R * self.RG))
                   / (1 + self.GRC_gamma * self.NG_R))

        self.NG_R = 1 + self.GRC_gamma * self.NG_R

    def get_reference(self):
        return self.RG

    def get_EG(self):
        return self.EG
