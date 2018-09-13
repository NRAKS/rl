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


class Bandit():
    def make_bandit(self, N_Act):
        probability = []
        for i in range(N_Act):
            probability.append(i / 10)
        return probability

    def make_liner_bandit(self, num_probability):
        probability = []
        for i in range(num_probability):
            for _ in range(num_probability - i):
                probability.append(i / 10)
        return probability


# 決定論的ツリーバンディット
class DTreeBandit(Environment):
    # 初期化,層の厚さを受け取る
    def __init__(self, layer, simulation_times=0, episode_times=0):
        super().__init__()
        self.layer = layer
        self.probability = np.asarray(
            [[0.4, 0.5, 0.6, 0.7], [0.9, 0.8, 0.1, 0.2],
             [0.7, 0.1, 0.3, 0.6], [0.6, 0.4, 0.1, 0.3],
             [0.5, 0.2, 0.3, 0.1], [0.3, 0.6, 0.5, 0.8]]
            )
        # 層の数からバンディットの数を計算
        self.num_bandit = 0
        self.num_state = 0

        for n in range(layer + 1):
            self.num_state += 4 ** n
            if n < layer:
                self.num_bandit += 4 ** n
            if n == layer-1:
                self.goal = self.num_bandit

        self.n_simu = simulation_times
        self.n_epi = episode_times

        # それぞれのバンディットに確率listからランダムに報酬期待値を当てはめる
        self.bandit = np.zeros((self.num_bandit, 4))
        for n in range(self.num_bandit):
            self.bandit[n] = random.choice(self.probability)

    # 次の状態判定
    def evaluate_next_state(self, current_state, current_action):
        self.next_state = 4 * current_state + current_action + 1

    # 報酬を返す
    def evaluate_reward(self, state, current_action):
        return self.bandit[[state], [current_action]]

    def get_goal_state(self):
        return self.goal

    def print_bandit(self):
        print("{}".format(self.bandit))


# 確率論的ツリーバンディット
class PTreeBandit(DTreeBandit):
    def __init__(self, layer, simulation_times=0, episode_times=0):
        super().__init__(layer, )

    # 報酬を返す
    def evaluate_reward(self, state, current_action):
        if random.random() <= self.bandit[[state], [current_action]]:
            return 1
        else:
            return 0


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


# モンティ・ホール問題
# 扉を選択→不正解の扉を見せられてから選択する扉を変更するかを選択
class MontyHall(Environment):
    def __init__(self, num_doors=3):
        super().__init__()
        self.num_doors = num_doors
        self.correct_answer = random.choice([0, 1, 2])
    
    # ドア選択を受け取って不正解の情報を見せる
    def evaluate_next_state(self, action):
        # 選んだドアを記録
        self.serect_door = action
        # ヤギ(不正解)のドアをオープンする
        goat = random.choice(list(set(self.num_doors) 
                             - set([self.correct_answer, action])))
        # ヤギのドア番号を返す
        return goat

    # ドアの変更選択した後の報酬処理
    def evaluate_reward(self, action):
        """
        action == 0 : そのまま
        action == 1 : 変更
        """
        # ドアスイッチの変更記録
        swich = action
        if action == 1:
            # 選んだドアの上書き
            self.serect_door = random.choice(list(set(self.num_doors) 
                                             - set([self.serect_door, self.goat])))
        
        if self.correct_answer == self.serect_door:
            reward = 1
        
        return [reward, swich]
