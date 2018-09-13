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
from collections import defaultdict


# TD学習アルゴリズム
class Q_learning(object):

    def __init__(self, learning_rate, discount_rate, num_state, num_action):
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.num_state = num_state
        self.num_action = num_action
        self.Q = defaultdict(int)

    def update_Q(
            self, current_state, next_state,
            current_action, reward, next_action=None):
        max_Q = max(self.Q[current_state, x] for x in range(self.num_action))
        TD = (reward
              + self.discount_rate
              * max_Q
              - self.Q[current_state, current_action])
        self.Q[current_state, current_action] += self.learning_rate * TD

    def init_params(self):
        self.Q = defaultdict(int)

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


# ε-greedy方策
class eps_greedy(object):
    def __init__(self, eps_init, eps_discount, eps_min, learning_rate, discount_rate, num_state, num_action, policy):
        self.eps = eps_init
        self.eps_init = eps_init
        self.eps_discount = eps_discount
        self.eps_min = eps_min

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
        if random.random() < self.eps:
            action = random.randint(0, len(Q[current_state])-1)

        else:
            action = np.argmax(Q[current_state])

        return action

    def update(self, current_state, next_state,
            current_action, reward, next_action=None):
        self.policy.update_Q(current_state, next_state,
                             current_action, reward, next_action)

    def update_eps(self):
        if self.eps > self.eps_min:
            self.eps -= self.eps_discount

    def init_params(self):
        self.policy.init_params()
        self.eps = self.eps_init


# RS(risk-sensitive satisficing)モデル
class RS(object):
    def __init__(
            self, learning_rate, discount_rate, reference,
            tau_alpha, tau_gamma, num_state, num_action, policy):
        self.reference_init = reference
        self.reference = defaultdict(lambda: reference)
        self.tau_alpha = tau_alpha
        self.tau_gamma = tau_gamma
        self.num_action = num_action
        self.num_state = num_state
        self.tau = defaultdict(int)
        self.tau_current = defaultdict(int)
        self.tau_post = defaultdict(int)

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
        Q = self.policy.get_Q()
        idx = defaultdict(int)
        for n_action in range(self.num_action):
            idx[n_action] = Q[next_state, n_action]
        action_up = max(idx, key=idx.get)
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
        self.reference = defaultdict(lambda: self.reference_init)
        self.tau = defaultdict(int)
        self.tau_current = defaultdict(int)
        self.tau_post = defaultdict(int)


class GRC(RS):
    def __init__(
            self, learning_rate, discount_rate, reference, zeta,
            tau_alpha, tau_gamma, num_action, num_state, policy):
        super().__init__(learning_rate, discount_rate, reference,
                         tau_alpha, tau_gamma, num_action, num_state, policy)
        self.RG = reference
        self.zeta = zeta
        self.EG = 0
        self.NG = 0
        self.GRC_gamma = discount_rate
        self.NG_R = 0

    def get_serect_action(self, current_state):
        DG = min([self.EG - self.RG, 0])
        Q = self.policy.get_Q()
        max_Q = max(Q[current_state, x] for x in range(self.num_action))
        reference = max_Q - self.zeta * DG
        rs = defaultdict(int)
        for n_action in range(self.num_action):
            rs[current_state, n_action] = (self.tau[current_state, n_action] 
                                * (Q[current_state, n_action]
                                - reference))
        idx = defaultdict(int)
        for n_action in range(self.num_action):
            idx[n_action] = rs[current_state, n_action]

        serect_action = max(idx, key=idx.get)

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
