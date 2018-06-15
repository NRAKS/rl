"""
python3
実験用タスクまとめクラス
基本的にやることは、
    環境設定に必要な情報を受け取り、初期設定
    行動と現状態を受け取り、次の状態を決定
    報酬を決定
    次状態などを返す
"""

import random
import numpy as np


class Environment(object):
    def __init__(self):
        self.Currentstate = 0
        self.NextState = 0
        self.reward = 0
        self.N_State = 0
        self.startstate = 0

    def GetNextState(self):
        return self.NextState

    def GetReward(self):
        return self.reward

    def GetNumState(self):
        return self.N_State

    def GetStartstate(self):
        return self.startstate
        
class Bandit():
    def Make_Bandit(self, N_Act):
        Pro = []
        for i in range(N_Act):
            Pro.append(i / 10)
        return Pro

    def Make_linerBandit(self, N_Pro):
        Pro = []
        for i in range(N_Pro):
            for n in range(N_Pro - i):
                Pro.append(i / 10)
        return Pro

#決定論的ツリーバンディット
class TreeBandit(Environment):
    #初期化,層の厚さを受け取る
    def __init__(self, layer, SimulationTimes=0, EpisodeTimes=0):
        super().__init__()
        self.layer = layer
        self.Pro = np.asarray(
            [[0.4, 0.5, 0.6, 0.7], [0.9, 0.8, 0.1, 0.2],
            [0.7, 0.1, 0.3, 0.6], [0.6, 0.4, 0.1, 0.3],
            [0.5, 0.2, 0.3, 0.1], [0.3, 0.6, 0.5, 0.8]]
            )
        #層の数からバンディットの数を計算
        self.N_Bandit = 0
        self.N_State = 0
        for n in range(layer+1):
            self.N_State += 4 ** n
            if n < layer:
                self.N_Bandit +=  4 ** n
            if n == layer-1:
                self.Goal = self.N_Bandit
        
        self.n_simu = SimulationTimes
        self.n_epi = EpisodeTimes

        #それぞれのバンディットに確率listからランダムに報酬期待値を当てはめる
        self.Bandit = np.zeros((self.N_Bandit, 4))
        for n in range(self.N_Bandit):
            self.Bandit[n] = random.choice(self.Pro)

    #次の状態判定
    def EvaluateNextState(self, Currentstate, CurrentAction):
        self.NextState = 4 * Currentstate + CurrentAction + 1
    
    #報酬を返す
    def EvaluateReward(self, State, CurrentAction):
        return self.Bandit[[State], [CurrentAction]]
    
    def GetGoalState(self):
        return self.Goal
    
    def PrintBandit(self):
        print("{}".format(self.Bandit))

#確率論的ツリーバンディット
class PTreeBandit(TreeBandit):
    def __init__(self, layer, SimulationTimes=0, EpisodeTimes=0):
        super().__init__(layer, SimulationTimes=0, EpisodeTimes=0)
    
    #報酬を返す
    def EvaluateReward(self, State, CurrentAction):
        if random.random() <= self.Bandit[[State], [CurrentAction]]:
            return 1
        else:
            return 0

class EasyMaze(Environment):
    #簡単なステージを作る
    def __init__(self, Row, Col, start, goal):
        super().__init__()
        self.row = Row
        self.col = Col
        self.start = start
        self.goal = goal

    #座標に変換
    def CoordToState(self, row, col):
        return ((row * self.col) + col)
    #座標からx軸を算出
    def StateToRow(self, state):
        return ((int)(state / self.col))
    #座標からy軸を算出
    def StateToCol(self, state):
        return (state % self.col)
    #次の座標を算出
    def EvaluateNextState(self, action, state):
        Upper = 0
        Lower = 1
        Left = 2
        Right = 3

        row = self.StateToRow(state)
        col = self.StateToCol(state)

        if action == Upper:
            if (row) > 0:
                row-=1
        elif action == Lower:
            if (row) < (self.row-1):
                row+=1
        elif action == Right:
            if (col)<(self.col-1):
                col+=1
        elif action == Left:
            if (col)>0:
                col-=1
                
        self.NextState = self.CoordToState(row, col)

    #報酬判定
    def EvaluateReward(self, state):
        if state == self.goal:
            return 1
        else:
            return 0

class Criff_world(Environment):
    #簡単なステージを作る
    def __init__(self, Row, Col, start, goal):
        super().__init__()
        self.row = Row
        self.col = Col
        self.start = start
        self.goal = goal

    #座標に変換
    def CoordToState(self, row, col):
        return ((row * self.col) + col)
    #座標からx軸を算出
    def StateToRow(self, state):
        return ((int)(state / self.col))
    #座標からy軸を算出
    def StateToCol(self, state):
        return (state % self.col)
    #次の座標を算出
    def EvaluateNextState(self, action, state):
        Upper = 0
        Lower = 1
        Left = 2
        Right = 3

        row = self.StateToRow(state)
        col = self.StateToCol(state)

        if action == Upper:
            if (row) > 0:
                row-=1
        elif action == Lower:
            if (row) < (self.row-1):
                row+=1
        elif action == Right:
            if (col)<(self.col-1):
                col+=1
        elif action == Left:
            if (col)>0:
                col-=1
                
        self.NextState = self.CoordToState(row, col)

    #報酬判定
    def EvaluateReward(self, state):
        if state == self.goal:
            
            return 1
            
        elif (self.row * (self.col -1) + 1 <= state and state <=  self.row * self.col - 2):
            
            return -10

        else:
            return 0