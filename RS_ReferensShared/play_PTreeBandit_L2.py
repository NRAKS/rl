"""
バンディットを終端状態にたどり着くまで行うタスク
複数のRSエージェントに満足化共有を行いながらタスクを学習させる
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import Task
import Policy
from SimuManager import SimulationManager

#シミュレーション設定
SimulationTimes = 10
EpisodeTimes = 5000
#タスク環境設定
Layer = 2
task = Task.PTreeBandit(Layer,SimulationTimes, EpisodeTimes)
print("バンディット確率表示")
task.PrintBandit()
StartState = 0
GoalState = task.GetGoalState()
n_Act = 4
n_state = task.GetNumState()
#エージェント設定(設定した数字の-1で実行される。)
N_agent = 4
R = 0
#R_simple = 0
alpha = 0.1
gamma = 1.0
Talpha = 0.1
Tgamma = 0.9
Ep = 1.0
DicEp = 0.005


def PlayTask():
    AveRewardlist = np.zeros((N_agent, EpisodeTimes))
    ave_rewardlist_per = np.zeros((N_agent, N_agent-1, EpisodeTimes))
    R_sharelist = np.zeros((N_agent, EpisodeTimes))
    for n_agent in range(3,N_agent):
        print("Agent num:{}".format(n_agent))
        Agent = SimulationManager(n_agent, SimulationTimes, EpisodeTimes, task)
        Agent.addRS(R, n_Act, n_state, alpha, gamma, Talpha, Tgamma, EpisodeTimes, SimulationTimes)

        for n_Simu in range(SimulationTimes):
            Agent.InitParameter()
            print("Simu:{}".format(n_Simu))
            for n_epi in range(EpisodeTimes):
                #print("Epi:{}".format(n_epi))
                Agent.play(n_epi)

        AveRewardlist[n_agent] = Agent.PlotAverageReward()
        
        R_sharelist[n_agent] = Agent.GetR_share() / SimulationTimes
        # for n in range(N_agent):
        ave_rewardlist_per[n_agent] = Agent.get_reward()

    with open(f"R_Sharelist_{Layer}L_{SimulationTimes}S_{EpisodeTimes}epi.pkl", mode="wb") as f:
        pkl.dump(R_sharelist, f)

    print("平均報酬獲得の結果表示")
    
    for n in range(1,N_agent):
        # plt.plot(AveRewardlist[n], label="Agent num:{}" .format(n))
        for j in range(len(ave_rewardlist_per[n])):
            plt.plot(ave_rewardlist_per[n,j], label="AgentGroup:{}, num:{}".format(n, j))
    plt.legend()
    plt.title("Reward Average")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    #plt.savefig("TreeBandit_RewardAve_{}L{}s{}e.png".format(Layer,SimulationTimes, EpisodeTimes))
    plt.show()
    plt.figure()

    print("Rの推移表示")
    for n in range(1, N_agent):
        plt.plot(R_sharelist[n], label = "Agent num:{}".format(n))
    plt.legend()
    plt.title("Reference time development")
    plt.xlabel("Episode")
    plt.ylabel("Reference")
    #plt.savefig("TreeBandit_Rtrans_{}L{}s{}e.png".format(Layer,SimulationTimes, EpisodeTimes))
    plt.show()
    plt.figure()   

PlayTask()