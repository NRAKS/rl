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

# 設定
NUM_ACTION = 4

BOARD_ROW = 7
BOARD_COL = 7
NUM_STATE = BOARD_COL * BOARD_ROW

START_STATE = (BOARD_ROW-1) * BOARD_COL
GOAL_STATE = (BOARD_COL * BOARD_ROW) - 1

SIMULAITON_TIMES = 1
EPISODE_TIMES = 500
reference = 0
learning_rate = 0.1
discount_rate = 0.9
tau_alpha = 0.1
tau_gamma = 0.9
eps = 0.0
discount_eps = 0
ZETA = 0.01

NUM_AGENT = 3

Maze = Task.CriffWorld(BOARD_ROW, BOARD_COL, START_STATE, GOAL_STATE)
player = [agent.GRC(learning_rate, discount_rate, reference,
                    ZETA, tau_alpha, tau_gamma, NUM_STATE, NUM_ACTION,
                    "Q_learning") for _ in range(NUM_AGENT)]


def play_task():

    sumreward_for_graph = np.zeros((NUM_AGENT, EPISODE_TIMES))
    for n_simu in range(SIMULAITON_TIMES):
        for i in range(NUM_AGENT):
            player[i].init_params()

        print("Simu:{}".format(n_simu))
        for n_epi in range(EPISODE_TIMES):

            sumreward = np.zeros(NUM_AGENT)
            for n in range(len(player)):
                current_state = START_STATE
                step = 0
                while True:
                    current_action = player[n].get_serect_action(current_state)
                    Maze.evaluate_next_state(current_action, current_state)
                    next_state = Maze.get_next_state()

                    reward = Maze.evaluate_reward(next_state)

                    sumreward[n] += reward
                    player[n].update(current_state, next_state,
                                     current_action, reward)
                    step += 1

                    if reward == 0:
                        current_state = Maze.get_next_state()
                    else:
                        current_state = START_STATE

                    if step == 100:
                        break

                sumreward_for_graph[n, n_epi] += sumreward[n]
                player[n].update_GRC_params(sumreward[n])
            R_update = np.average(sumreward)

            print("Episode : {}".format(n_epi))
            print("Etmp : {}".format(sumreward))
            print("R_update : {}".format(R_update))
            R_debug = np.zeros((NUM_AGENT))
            EG_debug = np.zeros((NUM_AGENT))
            for i in range(NUM_AGENT):
                # player[i].update_RG(R_update)
                player[i].update_GRC_reference(R_update)
                R_debug[i] = player[i].get_reference()
                EG_debug[i] = player[i].get_EG()
            print("EG値 : {}".format(EG_debug))
            print("RG値 : {}".format(R_debug))

    print("シミュレーション完了")
    for n in range(NUM_AGENT):
        plt.plot((sumreward_for_graph[n]) / SIMULAITON_TIMES,
                 label="RS_{}".format(n))
    plt.legend()
    plt.title("reward")
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.show()

play_task()
