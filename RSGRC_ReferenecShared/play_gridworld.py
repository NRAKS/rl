"""
python3
崖歩きタスク
複数のRS+GRCエージェントに満足化共有を行いながらタスクを学習させる
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import Task
import agent

#設定
n_act = 4

BoardRow = 7
BoardCol = 7
n_state = BoardCol * BoardRow

StartState = (BoardRow-1) * BoardCol
GoalState = (BoardCol * BoardRow) - 1

SimulationTimes = 1
EpisodeTimes = 500
R = 0
alpha = 0.1
gamma = 0.9
Talpha = 0.1
Tgamma = 0.9
Ep = 0.0
DicEp = 0
zeta = 0.01

n_agent = 3

Maze = Task.Criff_world(BoardRow, BoardCol, StartState, GoalState)
player = [agent.GRC(alpha, gamma, R, zeta, Talpha, Tgamma, n_act, n_state, "Q_learning") for _ in range(n_agent)]

def play_task():
    
    sumreward_for_graph = np.zeros((n_agent, EpisodeTimes))
    for n_simu in range(SimulationTimes):
        for i in range(n_agent):
            player[i].init_params()

        print("Simu:{}".format(n_simu))
        for n_epi in range(EpisodeTimes):
            
            sumreward = np.zeros(n_agent)
            for n in range(len(player)):
                current_state = StartState
                step = 0
                while True:
                    current_action = player[n].get_serect_action(current_state)
                    Maze.EvaluateNextState(current_action, current_state)
                    next_state = Maze.GetNextState()

                    reward = Maze.EvaluateReward(next_state)
                    
                    sumreward[n] += reward
                    player[n].update(current_state, next_state, current_action, reward)
                    step += 1

                    if reward == 0:
                        current_state = Maze.GetNextState()
                    else:
                        current_state = StartState
                        
                    if step == 100:
                        break
            
                sumreward_for_graph[n, n_epi] += sumreward[n]
                player[n].update_GRC_params(sumreward[n])
            R_update = np.average(sumreward)

            print("Episode : {}".format(n_epi))
            print("Etmp : {}".format(sumreward))
            print("R_update : {}".format(R_update))
            R_debug = np.zeros((n_agent))
            EG_debug = np.zeros((n_agent))
            for i in range(n_agent):
                #player[i].update_RG(R_update)
                player[i].update_GRC_reference(R_update)
                R_debug[i] = player[i].get_reference()
                EG_debug[i] = player[i].get_EG()
            print("EG値 : {}".format(EG_debug))
            print("RG値 : {}".format(R_debug))

    print("シミュレーション完了")
    for n in range(n_agent):
        plt.plot((sumreward_for_graph[n]) / SimulationTimes, label = "RS_{}".format(n))
    plt.legend()
    plt.title("reward")
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.show()

play_task()

