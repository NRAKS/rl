"""
２エージェントでオペラと映画どちらに行きたいか提案してもらう
以下の利得表に基づいた報酬を各エージェントに与える
------------------------
提案     opera    movie
opera   (3, 2)   (0, 0)
movie   (0, 0)   (2, 3)
------------------------
とりあえず獲得平均報酬Eiに対してε-greedyに行動してもらう？
1stepを1episodeとして何回かシミュレーションした平均をとる
"""

import random
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import TaskMultiAgent
import Policy


#シミュレーション設定
N_SIMULATION = 1000
N_EPISODE = 1000
#タスク環境設定
task = TaskMultiAgent.OperaMovie()
N_ACTION = task.get_numaction()
N_STATE = task.get_numstate()

#エージェント設定
N_AGENT = 2

alpha = 0.1
gamma = 0.9
Ep = 1.0
DicEp = Ep / 500

def playtask():
    agent = None
    agent = [Policy.Agent(alpha, gamma, N_ACTION, N_STATE, N_EPISODE, N_SIMULATION, Ep, DicEp) for _ in range(N_AGENT)]

    for n_simu in range(N_SIMULATION):
        for n in range(N_AGENT):
            agent[n].InitParameters()
        #print("Simu:{}".format(n_simu))
        sys.stdout.write("\r%s/%s" % (str(n_simu), str(N_SIMULATION-1)))
        for n in range(N_AGENT):
            agent[n].InitParameters()
        for n_epi in range(N_EPISODE):
            for n in range(N_AGENT):
                agent[n].InitState(task.get_startstate())
            #ここから行動開始

            while True:
                currentstate = agent[0].GetCurrentstate()
                for n in range(N_AGENT):
                    agent[n].SerectAction(agent[n].GetCurrentstate())
                task.evaluate_nextstate(agent[n].GetCurrentstate())
                nextstate = task.get_nextstate()
                task.evaluate_reward(agent[0].GetSerectAction(), agent[1].GetSerectAction())
                Reward_a, Reward_b = task.get_reward()

                agent[0].Update(agent[0].GetSerectAction(), currentstate, nextstate, Reward_a, n_epi)
                agent[1].Update(agent[1].GetSerectAction(), currentstate, nextstate, Reward_b, n_epi)
                
                if task.get_nextstate() >= task.get_goalstate():
                    break
                
                    

            
            
    print("平均報酬獲得の結果表示")
    for n in range(N_AGENT):
        plt.plot(agent[n].culculationAve(), label = "agent:{}".format(n))
    plt.ylim([0, 3])
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.show()

playtask()