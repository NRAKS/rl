"""
報酬期待値を返すバンディット問題
RSエージェントは最適基準と二人一組の２パターン
１ノードあたり４つの選択肢
層の深さを渡すと深さに応じたノードの数でタスクをこなすようにしたい
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import Task
import Policy
from SimuManager import SimulationManager

#シミュレーション設定
SimulationTimes = 1000
EpisodeTimes = 5000
#タスク環境設定
Layer = 5
task = Task.DTreeBandit(Layer,SimulationTimes, EpisodeTimes)
print("バンディット確率表示")
task.PrintBandit()
StartState = 0
GoalState = task.GetGoalState()
n_Act = 4
n_state = task.GetNumState()
#エージェント設定(設定した数字の-1で実行される。)
N_agent = 6
R = 0
#R_simple = 0
alpha = 0.1
gamma = 1.0
Talpha = 0.1
Tgamma = 0.9
Ep = 1.0
DicEp = 0.005

# Agent = SimulationManager(n_agent, Layer, SimulationTimes, EpisodeTimes, task)
# Agent.addRS(R, n_Act, n_state, alpha, gamma, Talpha, Tgamma, EpisodeTimes, SimulationTimes)
        
def SumIdealRewardperEpi(IdealReward, N_Epi):
    SumRewardperEpi = np.zeros((N_Epi))
    for i in range(N_Epi):
        if i >= 1:
            SumRewardperEpi[i] = IdealReward + SumRewardperEpi[i-1]
        else:
            SumRewardperEpi[i] = IdealReward
    return SumRewardperEpi

def Culculation_Rshare(Q1, Q2):
    maxQ = np.zeros((n_state))
    for state in range(n_state):
        maxQ[state] = max(max(Q1[state]), max(Q2[state]))
    return maxQ

def PlayTask():
    AveRewardlist = np.zeros((N_agent, EpisodeTimes))
    Regretlist = np.zeros((N_agent, EpisodeTimes))
    R_sharelist = np.zeros((N_agent, EpisodeTimes))
    for n_agent in range(1,N_agent):
        print("Agent num:{}".format(n_agent))
        Agent = SimulationManager(n_agent, SimulationTimes, EpisodeTimes, task)
        Agent.addRS(R, n_Act, n_state, alpha, gamma, Talpha, Tgamma, EpisodeTimes, SimulationTimes)
        IdealReward = task.SeekIdeal()
        print("最適報酬:{}".format(IdealReward))
        SumIdealReward = SumIdealRewardperEpi(IdealReward, EpisodeTimes)
        print("最適報酬和：{}".format(SumIdealReward))

        for n_Simu in range(SimulationTimes):
            Agent.InitParameter()
            print("Simu:{}".format(n_Simu))
            for n_epi in range(EpisodeTimes):
                #print("Epi:{}".format(n_epi))
                Agent.play(n_epi)

        AveRewardlist[n_agent] = Agent.PlotAverageReward()
        Regretlist[n_agent] = Agent.PlotRegret(SumIdealReward)
        R_sharelist[n_agent] = Agent.GetR_share() / SimulationTimes

    with open(f"R_Sharelist_{Layer}L_{SimulationTimes}S_{EpisodeTimes}epi.pkl", mode="wb") as f:
        pkl.dump(R_sharelist, f)

    print("平均報酬獲得の結果表示")
    Ideal = np.full(EpisodeTimes, IdealReward)
    for n in range(1,N_agent):
        plt.plot(AveRewardlist[n], label="Agent num:{}" .format(n))
    plt.plot(Ideal, label = "Optimum")
    plt.legend()
    #plt.title("output")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig("TreeBandit_RewardAve_{}L{}s{}e.png".format(Layer,SimulationTimes, EpisodeTimes))
    #plt.show()
    plt.figure()

    print("regret表示")
    for n in range(1,N_agent):
        #print("エージェント数:{}のグラフ出力".format(n))
        plt.plot(Regretlist[n], label="Agent num:{}" .format(n))
    plt.legend()
    #plt.title("regret output")
    plt.xlabel("Episode")
    plt.ylabel("regret")
    plt.savefig("TreeBandit_regret_{}L{}s{}e.png".format(Layer,SimulationTimes, EpisodeTimes))
    plt.figure()
    #plt.show()

    # print("Rの推移表示")
    # for n in range(1, N_agent):
    #     plt.plot(R_sharelist[n], label = "Agent num:{}".format(n))
    # plt.legend()
    # plt.xlabel("Episode")
    # plt.ylabel("Reference")
    # plt.savefig("TreeBandit_Rtrans_{}L{}s{}e.png".format(Layer,SimulationTimes, EpisodeTimes))
    # plt.figure()
    

PlayTask()