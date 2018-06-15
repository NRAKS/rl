"""
python3
決定論的ツリーバンディットタスク
複数のRS+GRCエージェントに満足化共有を行いながらタスクを学習させる
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import Task
import agent


#設定
n_act = 4

layer = 2
StartState = 0

SimulationTimes = 1
EpisodeTimes = 5000
R = 0.12
alpha = 0.1
gamma = 0.9
Talpha = 0.1
Tgamma = 0.9
Ep = 0.0
DicEp = 0
zeta = 0.01

n_agent = 3
task = Task.TreeBandit(layer, SimulationTimes, EpisodeTimes)
n_state = task.GetNumState()
GetGoalState = task.GetGoalState()

player = [agent.GRC(alpha, gamma, R, zeta, Talpha, Tgamma, n_act, n_state, "Q_learning") for _ in range(n_agent)]

print("バンディットの確率表示")
task.PrintBandit()

def play_task():
    
    sumreward_for_graph = np.zeros((n_agent, EpisodeTimes))
    for n_simu in range(SimulationTimes):
        for i in range(n_agent):
            player[i].init_params()

        EG_graph = np.zeros((n_agent, EpisodeTimes))
        RG_graph = np.zeros((n_agent, EpisodeTimes))

        print("Simu:{}".format(n_simu))
        for n_epi in range(EpisodeTimes):
            
            sumreward = np.zeros(n_agent)

            for n in range(len(player)):
                current_state = StartState
                step = 0
                while True:
                    current_action = player[n].get_serect_action(current_state)

                    task.EvaluateNextState(current_state, current_action)

                    next_state = task.GetNextState()

                    reward = task.EvaluateReward(current_state, current_action)
                    
                    sumreward[n] += reward
                    player[n].update(current_state, next_state, current_action, reward)
                    step += 1

                    
                    current_state = task.GetNextState()
                        
                    if current_state >= task.GetGoalState():
                        break
            
                sumreward_for_graph[n, n_epi] += sumreward[n]
                player[n].update_GRC_params(sumreward[n])
                EG_graph[n, n_epi] = player[n].get_EG()
                RG_graph[n, n_epi] =player[n].get_reference()
            R_update = np.average(sumreward)
            R_debug = np.zeros((n_agent))
            EG_debug = np.zeros((n_agent))
            for i in range(n_agent):
                #player[i].update_RG(R_update)
                player[i].update_GRC_reference(R_update)
                R_debug[i] = player[i].get_reference()
                EG_debug[i] = player[i].get_EG()
            if n_epi == EpisodeTimes-1:
                print("Episode : {}".format(n_epi))
                print("Etmp : {}".format(sumreward))
                print("R_update : {}".format(R_update))
                print("EG値 : {}".format(EG_debug))
                print("RG値 : {}".format(R_debug))

    print("シミュレーション完了")
    print("バンディットの確率表示")
    task.PrintBandit()

    print("獲得平均報酬")
    for n in range(n_agent):
        plt.plot((sumreward_for_graph[n]) / SimulationTimes, label = "RS_{}".format(n))
    plt.legend()
    plt.title("reward")
    plt.xlabel("episode")
    plt.ylabel("reward")
    #plt.savefig("Sumreward_ave_time_development_Simu{}_Epi{}_Agent{}".format(SimulationTimes, EpisodeTimes, n_agent))
    plt.show()
    #plt.figure()

    print("EGの時間発展")
    for n in range(n_agent):
        plt.plot(EG_graph[n], label = "RS_{}".format(n))
    plt.legend()
    plt.title("EG time development")
    plt.xlabel("episode")
    plt.ylabel("EG")
    #plt.savefig("EG_time_development_Simu{}_Epi{}_Agent{}".format(SimulationTimes, EpisodeTimes, n_agent))
    plt.show()
    #plt.figure()

    print("RGの時間発展")
    for n in range(n_agent):
        plt.plot(RG_graph[n], label = "RS_{}".format(n))
    plt.legend()
    plt.title("RG time development")
    plt.xlabel("episode")
    plt.ylabel("RG")
    #plt.savefig("RG_time_development_Simu{}_Epi{}_Agent{}".format(SimulationTimes, EpisodeTimes, n_agent))
    plt.show()
    #plt.figure()

play_task()

