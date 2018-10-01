"""
python3
実験用タスクまとめクラス
基本的にやることは、
    環境設定に必要な情報を受け取り、初期設定
    行動と現状態を受け取り、次の状態を決定
    報酬を決定
    次状態などを返す
"""
import agent
import random
import numpy as np
import copy


class Environment(object):
    def __init__(self):
        self.current_state = 0
        self.next_state = 0
        self.reward = 0
        self.num_state = 0
        self.start_state = 0

    def get_current_state(self):
        return self.current_state 

    def get_next_state(self):
        return self.next_state

    def get_reward(self):
        return self.reward

    def get_num_state(self):
        return self.num_state

    def get_start_state(self):
        return self.start_state



class EasyMaze(Environment):
    # 簡単なステージを作る
    def __init__(self, row, col, start, goal):
        super().__init__()
        self.row = row
        self.col = col
        self.start = start
        self.goal = goal

    # 座標に変換
    def coord_to_state(self, row, col):
        return ((row * self.col) + col)

    # 座標からx軸を算出
    def state_to_row(self, state):
        return ((int)(state / self.col))

    # 座標からy軸を算出
    def state_to_col(self, state):
        return (state % self.col)

    # 次の座標を算出
    def evaluate_next_state(self, action, state):
        UPPER = 0
        LOWER = 1
        LEFT = 2
        RIGHT = 3
        STOP = 4

        row = self.state_to_row(state)
        col = self.state_to_col(state)

        if action == UPPER:
            if (row) > 0:
                row -= 1
        elif action == LOWER:
            if (row) < (self.row-1):
                row += 1
        elif action == RIGHT:
            if (col) < (self.col-1):
                col += 1
        elif action == LEFT:
            if (col) > 0:
                col -= 1
        elif action == STOP:
            pass

        self.next_state = self.coord_to_state(row, col)

    # 報酬判定
    def evaluate_reward(self, state):
        if state == self.goal:
            return 1
        else:
            return 0


# 崖歩きタスク
class CriffWorld(EasyMaze):
    def __init__(self, row, col, start, goal):
        super().__init__()
        self.row = row
        self.col = col
        self.start = start
        self.goal = goal

    # 報酬判定
    def evaluate_reward(self, state):
        if state == self.goal:
            return 1

        elif (self.row * (self.col - 1) + 1 <= state
              and state <= self.row * self.col - 2):
            return -10

        else:
            return 0


# 収穫タスク
# 特定のマスに居続けて収穫を得て、拠点に持ち帰る
class HarvestWorld(EasyMaze):
        def __init__(self, row, col, start, harvest_state1=None, harvest_state2=None):
            super().__init__(row, col, start, goal=None)
            self.row = row
            self.col = col
            self.start = start
            self.harvest_state1 = harvest_state1
            self.harvest_state2 = harvest_state2

        def evaluate_next_state(self, action, current_state, n_agent):
            super().evaluate_next_state(action, current_state[n_agent])
            next_state = copy.deepcopy(current_state)
            next_state[n_agent] = self.next_state
            self.next_state = next_state

        def evaluate_reward(self, state):
            if state == self.harvest_state1:
                return 10
            elif state == self.harvest_state2:
                return 5
            else:
                return 0
