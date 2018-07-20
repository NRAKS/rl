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


class Environment(object):
    def __init__(self):
        self.current_state = 0
        self.next_state = 0
        self.reward = 0
        self.num_state = 0
        self.start_state = 0

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

    # 生成されたツリーバンディットの最適報酬を求める
    def seek_ideal(self):
        n_Act = 4
        n_state = self.get_num_state()
        EpisodeTimes = self.n_epi
        SimulationTimes = self.n_simu
        Agent = Policy.MinTrial(n_Act, n_state, EpisodeTimes, SimulationTimes)
        for _ in range(4**(self.layer+1)):
            CurrentState = self.start_state
            while True:
                Agent.SerectAction(CurrentState)
                self.evaluate_next_state(CurrentState, Agent.GetSerectAction())
                NextState = self.get_next_state()
                Reward = self.evaluate_reward(CurrentState,
                                              Agent.GetSerectAction())
                Agent.Update(Reward)
                CurrentState = NextState
                if CurrentState >= self.get_goal_state():
                    Agent.InitSumReward()
                    break

        return Agent.GetMaxReward()


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
    def __init__(self, Row, Col, start, goal):
        super().__init__()
        self.row = Row
        self.col = Col
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

        self.next_state = self.coord_to_state(row, col)

    # 報酬判定
    def evaluate_reward(self, state):
        if state == self.goal:
            return 1
        else:
            return 0


class CriffWorld(EasyMaze):
    def __init__(self, Row, Col, start, goal):
        super().__init__(Row, Col, start, goal)

    # 報酬判定
    def evaluate_reward(self, state):
        if state == self.goal:
            return 1

        elif (self.row * (self.col - 1) + 1 <= state
              and state <= self.row * self.col - 2):
            return -10

        else:
            return 0
