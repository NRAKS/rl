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


class Greedy(object):  # greedy方策
    # 行動価値を受け取って行動番号を返す
    def serect_action(self, value, current_state):
        return np.argmax(value[current_state])
    
    def init_params(self):
        pass

    def update_params(self):
        pass


class EpsGreedy(Greedy):
    def __init__(self, eps):
        self.eps = eps

    def serect_action(self, value, current_state):
        if random.random() < self.eps:
            return random.choice(range(len(value[current_state])))

        else:
            return np.argmax(value[current_state])


class EpsDecGreedy(EpsGreedy):
    def __init__(self, eps, eps_min, eps_decrease):
        super().__init__(eps)
        self.eps_init = eps
        self.eps_min = eps_min
        self.eps_decrease = eps_decrease

    def init_params(self):
        self.eps = self.eps_init

    def update_params(self):
        self.eps -= self.eps_decrease


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



# RS(risk-sensitive satisficing)モデル
class RS(object):
    def __init__(
            self, learning_rate, discount_rate, reference,
            tau_alpha, tau_gamma, num_state, num_action, value_func):
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
        if value_func == "Q_learning":
            self.policy = Q_learning(learning_rate, discount_rate,
                                     num_state, num_action)
        elif value_func == "sarsa":
            self.policy = Sarsa(learning_rate, discount_rate,
                                num_state, num_action)
        else:
            sys.exit("Error")

    def serect_action(self, current_state): 
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

    def get_reference(self):
        return self.reference

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

    def serect_action(self, current_state):
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


# エージェントクラス
class Agent():
    def __init__(self, value_func="Q_learning", policy="greedy", learning_rate=0.1, discount_rate=0.9, eps=None, eps_min=None, eps_decrease=None, n_state=None, n_action=None):
        # 価値更新方法の選択
        if value_func == "Q_learning":
            self.value_func = Q_learning(num_state=n_state, num_action=n_action)
        
        else:
            print("error:価値関数候補が見つかりませんでした")
            sys.exit()

        # 方策の選択
        if policy == "greedy":
            self.policy = Greedy()
        
        elif policy == "eps_greedy":
            self.policy = EpsGreedy(eps=eps)

        elif policy == "eps_dec_greedy":
            self.policy = EpsDecGreedy(eps=eps, eps_min=eps_min, eps_decrease=eps_decrease)

        else:
            print("error:方策候補が見つかりませんでした")
            sys.exit()

    # パラメータ更新(基本呼び出し)
    def update(self, current_state, current_action, reward, next_state):
        self.value_func.update_Q(current_state, current_action, reward, next_state)
        self.policy.update_params()

    # 行動選択(基本呼び出し)
    def serect_action(self, current_state):
        return self.policy.serect_action(self.value_func.get_Q(), current_state)

    # 行動価値の表示
    def print_value(self):
        print(self.value_func.get_Q())

    # 所持パラメータの初期化
    def init_params(self):
        self.value_func.init_params()
        self.policy.init_params()
