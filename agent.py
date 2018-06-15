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

#TD学習アルゴリズム   
class Q_learning(object):
    
    def __init__(self, alpha, gamma, n_state, n_act):
        self.alpha = alpha
        self.gamma = gamma
        self.n_state = n_state
        self.n_act = n_act
        self.Q = np.zeros((n_state, n_act))

    def update_Q(self, current_state, next_state, current_action, reward, next_action = None):
        maxQ = max(self.Q[next_state])
        TD = reward + self.gamma * maxQ - self.Q[current_state, current_action]
        self.Q[current_state, current_action] += self.alpha * TD
    
    def init_params(self):
        self.Q = np.zeros((self.n_state, self.n_act))

    def get_Q(self):
        return self.Q

class sarsa(Q_learning):
    def __init__(self, alpha, gamma, n_state, n_act):
        super.__init__(alpha, gamma, n_state, n_act)

    def update_Q(self, current_state, next_state, current_action, reward, next_action):
        next_Q = self.Q[next_state, next_action]
        TD = reward + self.gamma * next_Q - self.Q[current_state, current_action]
        self.Q[current_state, current_action] += self.alpha * TD

#RS(risk-sensitive satisficing)モデル
class RS(object):
    def __init__(self, alpha, gamma, R, tau_alpha, tau_gamma, n_act, n_state, policy):
        self.r_init = R
        self.R = np.full(n_state, R)
        self.tau_alpha = tau_alpha
        self.tau_gamma = tau_gamma
        self.n_act = n_act
        self.n_state = n_state
        self.tau = np.zeros((n_state, n_act))
        self.tau_current = np.zeros((n_state, n_act))
        self.tau_post = np.zeros((n_state, n_act))

        #方策によってQ学習とsarsaを切り替える
        if policy ==  "Q_learning":
            self.policy = Q_learning(alpha, gamma, n_state, n_act)
        elif policy == "sarsa":
            self.policy = sarsa(alpha, gamma, n_state, n_act)
        else:
            sys.exit("Error")

    def get_serect_action(self, current_state): 
        Q = self.policy.get_Q()
        rs = self.tau[current_state] * (Q[current_state] - self.R[current_state])

        idx = np.where(rs == max(rs))
        serect_action = random.choice(idx[0])
        return serect_action

    def update(self, current_state, next_state, current_action, reward, next_action = None):
        self.policy.update_Q(current_state, next_state, current_action, reward, next_action)
        #τ値更新準備
        max_next_state_Q = max(self.policy.get_Q()[next_state])
        idx = np.where(self.policy.get_Q()[next_state] == max_next_state_Q)
        action_up = random.choice(idx[0])
        #τ値更新
        self.tau_current[current_state, current_action] += 1
        self.tau_post[current_state, current_action] += self.tau_alpha * (self.tau_gamma * self.tau[next_state, action_up] - self.tau_post[current_state, current_action])

        self.tau[current_state, current_action] = self.tau_current[current_state, current_action] + self.tau_post [current_state, current_action]
        
    def init_params(self):
        self.policy.init_params()
        self.R = np.full(self.n_state, self.r_init)
        self.tau = np.zeros((self.n_state, self.n_act))
        self.tau_current = np.zeros((self.n_state, self.n_act))
        self.tau_post = np.zeros((self.n_state, self.n_act))

class GRC(RS):
    def __init__(self, alpha, gamma, R, zeta, tau_alpha, tau_gamma, n_act, n_state, policy):
        super().__init__(alpha, gamma, R, tau_alpha, tau_gamma, n_act, n_state, policy)
        self.RG = R
        self.zeta = zeta
        self.EG = 0
        self.NG = 0
        self.GRC_gamma = gamma
        self.NG_R = 0

    
    def get_serect_action(self, current_state):
        DG = min([self.EG - self.RG, 0])
        Q = self.policy.get_Q()
        max_Q = max(Q[current_state])
        r = max_Q - self.zeta * DG
        rs = {}
        rs[current_state] = self.tau[current_state] * (Q[current_state] - r) 
        idx = np.where(rs[current_state] == max(rs[current_state]))

        serect_action = random.choice(idx[0])

        return serect_action

    def update_GRC_params(self, sumreward):
        self.Etmp = sumreward
        self.EG = (self.Etmp + self.GRC_gamma * (self.NG * self.EG)) / (1 + self.GRC_gamma * self.NG)
        self.NG = 1 + self.GRC_gamma * self.NG

    def init_params(self):
        super().init_params()
        self.EG = 0
        self.NG = 0
    
    def update_RG(self, RG_up):
        self.RG = RG_up
        
    #実行してもいい結果は得られなかった
    def update_GRC_reference(self, R_up):
        self.Rtmp = R_up
        self.RG = (self.Rtmp + self.GRC_gamma * (self.NG_R * self.RG)) / (1 + self.GRC_gamma * self.NG_R)
        self.NG_R = 1 + self.GRC_gamma * self.NG_R
        
    def get_reference(self):
        return self.RG
        
    def get_EG(self):
        return self.EG