"""
複数エージェント用の強化学習タスクまとめ
基本的に
    環境構築に必要な情報受け取り、初期設定
    現状態と行動から、次の状態決定
    報酬を決定

    持ってる情報は
    スタート地点
    ゴール地点
    報酬
    状態
    取れる行動数
"""
import Policy
import random
import numpy as numpy

#基本的な機能
class Enviroment(object):
    def __init__(self):
        self.currentstate = 0
        self.nextstate = 0
        self.reward = 0
        self.n_state = 0
        self.startstate = 0
        self.goalstate = 0
        self.n_action = 0

    def get_nextstate(self):
        return self.nextstate

    def get_reward(self):
        return self.reward

    def get_numstate(self):
        return self.n_state

    def get_startstate(self):
        return self.startstate
    
    def get_goalstate(self):
        return self.goalstate

    def get_numaction(self):
        return self.n_action


#２エージェントでオペラか映画のどちらに行きたいかを提案、組み合わせに応じた報酬を返す
class OperaMovie(Enviroment):
    def __init__(self):
        super().__init__()
        #エージェントaが受け取る報酬
        self.reward_a = 0
        #エージェントbが受け取る報酬
        self.reward_b = 0
        self.n_action = 2
        self.startstate = 0
        self.goalstate = 1
        self.n_state = 2

    def evaluate_nextstate(self, currentstate):
        self.nextstate = 1
    
    def evaluate_reward(self, currentaction_a, currentaction_b):
        OPERA = 0
        MOVIE = 1
        if currentaction_a == currentaction_b == OPERA:
            self.reward_a = 3
            self.reward_b = 2
        elif currentaction_a == currentaction_b == MOVIE:
            self.reward_a = 2
            self.reward_b = 3
        else:
            self.reward_a = self.reward_b = 0
    
    def get_reward(self):
        return self.reward_a, self.reward_b
            

