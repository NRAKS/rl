"""
python3
方策まとめクラス
"""

import random
import numpy as np
import math

# Q学習アルゴリズム
class Agent(object):
    
    def __init__(self, Alpha, Gamma, N_Act, N_state, N_Epi, N_Simu, Ep=0, DicreaseEp=0):
        self.Q = np.zeros((N_state, N_Act))
        self.alpha = Alpha
        self.gamma = Gamma
        self.N_Act = N_Act
        self.N_state = N_state
        self.N_Epi = N_Epi
        self.N_Simu = N_Simu
        self.Ep = Ep
        self.Ep_init = Ep
        self.DicEp = DicreaseEp
        self.SumRewardperEpi = np.zeros((N_Epi))    #list...グラフ生成用
        self.Sumreward = np.zeros((N_Epi))
        self.Sumreward_Epi = 0  #
        self.step = 0
        self.SumStep = np.zeros((N_Epi))
        self.SerectAct = 0
        self.Currentstate = 0

    # Q値の更新
    def UpdateQ(self, CurrentState, NextState, CurrentAction, Reward):
        MaxvalueQ = max(self.Q[NextState, x] for x in range(self.N_Act))
        self.Q[CurrentState, CurrentAction] += self.alpha * (Reward + self.gamma * MaxvalueQ - self.Q[CurrentState, CurrentAction])

    # εの減衰
    def DicreaseEp(self):
        if self.Ep >= 0:
            self.Ep -= self.DicEp

    # 行動選択
    def SerectAction(self, CurrentState):
        if random.random()< self.Ep:
            SerectAction = random.randint(0, self.N_Act-1)
        else:
            _Q = np.asarray([self.Q[CurrentState, x]for x in range(self.N_Act)])
            # print(f"Q:{_Q}")
            idx = np.where(_Q == max(_Q))
            # print("Currentstate:{}".format(CurrentState))
            # print("idx: {}".format(idx[0]))
            SerectAction = random.choice(idx[0])
            # print(f"serectaction:{SerectAction}")
        self.SerectAct = SerectAction

    # Q値を返す
    def GetQ(self):
        return self.Q

    # 行動番号を返す
    def GetSerectAction(self):
        return self.SerectAct

    # 現在値を返す
    def GetCurrentstate(self):
        return self.Currentstate

    # ステップ数をカウント
    def CountStep(self, n_epi):
        self.SumStep[n_epi] += 1

    # 報酬をカウント
    def CountReward(self, reward, n_epi):
        self.SumRewardperEpi[n_epi] += reward
        self.Sumreward_Epi += reward

    # エピソードごとの報酬合計値をカウント
    def CountRsub(self, n_epi):
        self.Sumreward[n_epi] += self.Sumreward_Epi

    # 報酬の合計値平均を返す
    def GetSumReward(self):
        return self.Sumreward/self.N_Simu

    # 合計ステップ数を返す
    def GetSumStep(self):
        return self.SumStep

    # 平均値計算して返す
    def culculationAve(self):
        return self.SumRewardperEpi/self.N_Simu

    # 各種パラメータの初期化
    def InitParameters(self):
        self.Q = np.zeros((self.N_state, self.N_Act))
        self.Ep = self.Ep_init
        self.Sumreward_Epi = 0
        # self.SumStep = np.zeros((self.N_Epi))

    # 現在値を初期化する
    def InitState(self,state):
        self.Currentstate = state

    # 各種値の更新
    def Update(self, CurrentAction, CurrentState, NextState, Reward, n_epi):
        # Q値の更新
        self.UpdateQ(CurrentState, NextState, CurrentAction, Reward)
        # エピソードごとの報酬の合計値を更新
        self.CountReward(Reward, n_epi)
        # ステップ更新
        self.CountStep(n_epi)

    def Updatestate(self, NextState):
        self.Currentstate = NextState


# RSアルゴリズム
class RS(Agent):
    def __init__(self, R, N_Act, N_state, Alpha, Gamma, TAlpha, TGamma, N_Epi, N_Simu):
        super().__init__(Alpha, Gamma, N_Act, N_state, N_Epi, N_Simu)
        self.R = np.full(N_state, R)
        self.Rinit = R
        self.T = np.zeros((N_state, N_Act))
        self.Tcurrent = np.zeros((N_state, N_Act))
        self.Tpost = np.zeros((N_state, N_Act))
        self.TAlpha = TAlpha
        self.TGamma = TGamma

    # 行動選択
    def SerectAction(self, state):
        rs = np.zeros((self.N_Act))
        for act in range(self.N_Act):
            # print("T:{}".format(self.T))
            # print("Q:{}".format(self.Q))
            # print("R:{}".format(self.R))
            rs[act] = self.T[state, act] * (self.Q[state, act] - self.R[state])
        idx = np.where(rs == max(rs))
        idx_Q = np.where(self.Q[state] == max(self.Q[state]))
        SerectAction = random.choice(idx[0])
        if SerectAction in idx_Q[0]:
            greedy = 1
        self.SerectAct = SerectAction

    # τ値更新
    def Tupdate(self, CurrentAction, CurrentState, NextState):
        # t値を更新(準備)
        NextStateValueQ = np.asarray([self.Q[NextState, x] for x in range(self.N_Act)])
        MaxValueQ = max(self.Q[NextState, x] for x in range(self.N_Act))
        idx = np.where(NextStateValueQ == MaxValueQ)
        ActionUP = random.choice(idx[0])
        # τ値３種類を更新
        self.Tcurrent[CurrentState, CurrentAction] += 1

        self.Tpost[CurrentState, CurrentAction] += self.TAlpha * (self.TGamma * self.T[NextState, ActionUP] - self.Tpost[CurrentState, CurrentAction])

        self.T[CurrentState, CurrentAction] = self.Tcurrent[CurrentState, CurrentAction] + self.Tpost[CurrentState, CurrentAction]

    # 各種パラメータ初期化
    def InitParameters(self):
        super().InitParameters()
        self.R = np.full(self.N_state, self.Rinit)
        self.T = np.zeros((self.N_state, self.N_Act))
        self.Tcurrent = np.zeros((self.N_state, self.N_Act))
        self.Tpost = np.zeros((self.N_state, self.N_Act))

    # R値の共有
    def ShareReference(self, R_share):
        self.R = R_share
        
    # 行動から各種値の更新
    def Update(self, CurrentAction, CurrentState, NextState, Reward, n_epi):
        # T値の更新
        super().Update(CurrentAction, CurrentState, NextState, Reward, n_epi)
        self.Tupdate(CurrentAction, CurrentState, NextState)
    

class UCB1_tuned(Agent):
    def __init__(self, N_Act, N_state, Alpha, Gamma, N_Epi, N_Simu):
        super().__init__(Alpha, Gamma, N_Act, N_state, N_Epi, N_Simu)

        self.T = np.zeros((N_state, N_Act))

    def SerectAction(self, CurrentState):
        UCB = np.zeros((self.N_Act))
        T_sum = sum(self.T[CurrentState])
        # print("currentstate:{}, T:{}, min:{}".format(CurrentState, self.T[CurrentState], min(self.T[CurrentState])))
        if min(self.T[CurrentState] != 0):
            Q_sum = sum(self.Q[CurrentState])
            Q_Ave = Q_sum / T_sum
            QQ_sum = sum(self.Q[CurrentState] * self.Q[CurrentState])
            # print("Q_sum:{}, Q_Ave:{}, QQ_sum:{}".format(Q_sum, Q_Ave, QQ_sum))
            for i in range(self.N_Act):
                # print("i:{}".format(i))
                Vi = (QQ_sum/self.T[CurrentState, i]) - (Q_Ave ** 2) + math.sqrt((2 * math.log(T_sum)) / self.T[CurrentState, i])
                # print("vi:{}".format(Vi))
                UCB[i] = self.Q[CurrentState, i] + math.sqrt((math.log(T_sum) / self.T[CurrentState, i]) * min(1/4, Vi))
            idx = np.where(UCB == max(UCB))
            SerectAct = random.choice(idx[0])
        else:
            idx = np.where(self.T[CurrentState] == 0)
            SerectAct = random.choice(idx[0])
        self.T[CurrentState, SerectAct] += 1
        self.SerectAct = SerectAct

    def InitParameters(self):
        super().InitParameters()
        self.T = np.zeros((self.N_state, self.N_Act))


# 試行回数が一番少ないものを選び続ける(最適合計報酬を求める用)
class MinTrial(Agent):
    def __init__(self, N_Act, N_state, N_Epi, N_Simu):
        super().__init__(0, 0, N_Act, N_state, N_Epi, N_Simu)
        self.Trial = np.zeros((N_state, N_Act))
        self.MaxReward = 0
        self.SumReward = 0

    def SerectAction(self, CurrentState):
        idx = np.where(self.Trial[CurrentState] == min(self.Trial[CurrentState]))
        self.SerectAct = random.choice(idx[0])
        self.Trial[CurrentState, self.SerectAct] += 1

    def CountReward(self, Reward):
        self.SumReward += Reward

    def Update(self,Reward):
        self.CountReward(Reward)
        if self.SumReward > self.MaxReward:
            self.MaxReward = self.SumReward
    
    def InitParameters(self):
        self.Trial = np.zeros((self.N_state, self.N_Act))
        self.MaxReward = 0
        self.SumReward

    def InitSumReward(self):
        self.SumReward = 0
    
    def GetMaxReward(self):
        return self.MaxReward
