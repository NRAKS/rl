"""
python3
確率的ツリーバンディットタスク
複数のRS+GRCエージェントに満足化共有を行いながらタスクを学習させる
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import Task
import agent


# 設定
NUM_ACTION = 4
LAYER = 2
START_STATE = 0

SIMULATION_TIMES = 1
EPISODE_TIMES = 5000
reference = 0.0
learning_rate = 0.1
discount_rate = 0.9
tau_alpha = 0.1
tau_gamma = 0.9
Ep = 0.0
discountEp = 0
ZETA = 0.01

NUM_AGENT = 3
task = Task.PTreeBandit(LAYER, SIMULATION_TIMES, EPISODE_TIMES)
NUM_STATE = task.get_num_state()
GOAL_STATE = task.get_goal_state()

player = [agent.GRC(learning_rate, discount_rate, reference,
                    ZETA, tau_alpha, tau_gamma, NUM_STATE, NUM_ACTION,
                    "Q_learning") for _ in range(NUM_AGENT)]

print("バンディットの確率表示")
task.print_bandit()


def play_task():

    sumreward_for_graph = np.zeros((NUM_AGENT, EPISODE_TIMES))
    for n_simu in range(SIMULATION_TIMES):
        for i in range(NUM_AGENT):
            player[i].init_params()

        EG_graph = np.zeros((NUM_AGENT, EPISODE_TIMES))
        RG_graph = np.zeros((NUM_AGENT, EPISODE_TIMES))

        print("Simu:{}".format(n_simu))
        for n_epi in range(EPISODE_TIMES):

            sum_reward = np.zeros(NUM_AGENT)

            for n in range(len(player)):
                current_state = START_STATE
                step = 0

                while True:
                    current_action = player[n].get_serect_action(current_state)

                    task.evaluate_next_state(current_state, current_action)

                    next_state = task.get_next_state()

                    reward = task.evaluate_reward(current_state,
                                                  current_action)

                    sum_reward[n] += reward
                    player[n].update(current_state, next_state,
                                     current_action, reward)
                    step += 1

                    current_state = task.get_next_state()

                    if current_state >= task.get_goal_state():
                        break

                sumreward_for_graph[n, n_epi] += sum_reward[n]
                player[n].update_GRC_params(sum_reward[n])
                EG_graph[n, n_epi] = player[n].get_EG()
                RG_graph[n, n_epi] = player[n].get_reference()
            R_update = np.average(sum_reward)
            R_debug = np.zeros((NUM_AGENT))
            EG_debug = np.zeros((NUM_AGENT))
            for i in range(NUM_AGENT):
                # player[i].update_RG(R_update)
                player[i].update_GRC_reference(R_update)
                R_debug[i] = player[i].get_reference()
                EG_debug[i] = player[i].get_EG()
            if n_epi == EPISODE_TIMES - 1:
                print("Episode : {}".format(n_epi))
                print("Etmp : {}".format(sum_reward))
                print("R_update : {}".format(R_update))
                print("EG値 : {}".format(EG_debug))
                print("RG値 : {}".format(R_debug))

    print("シミュレーション完了")
    print("バンディットの確率表示")
    task.print_bandit()

    print("獲得平均報酬")
    for n in range(NUM_AGENT):
        plt.plot((sumreward_for_graph[n]) / SIMULATION_TIMES,
                 label="RS_{}".format(n))
    plt.legend()
    plt.title("reward")
    plt.xlabel("episode")
    plt.ylabel("reward")
    # plt.savefig("Sumreward_ave_time_development_Simu{}_Epi{}_Agent{}"
    #             .format(SimulationTimes, EpisodeTimes, n_agent))
    plt.show()
    # plt.figure()

    print("EGの時間発展")
    for n in range(NUM_AGENT):
        plt.plot(EG_graph[n], label="RS_{}".format(n))
    plt.legend()
    plt.title("EG time development")
    plt.xlabel("episode")
    plt.ylabel("EG")
    # plt.savefig("EG_time_development_Simu{}_Epi{}_Agent{}"
    #             .format(SimulationTimes, EpisodeTimes, n_agent))
    plt.show()
    # plt.figure()

    print("RGの時間発展")
    for n in range(NUM_AGENT):
        plt.plot(RG_graph[n], label="RS_{}".format(n))
    plt.legend()
    plt.title("RG time development")
    plt.xlabel("episode")
    plt.ylabel("RG")
    # plt.savefig("RG_time_development_Simu{}_Epi{}_Agent{}"
    #             .format(SimulationTimes, EpisodeTimes, n_agent))
    plt.show()
    # plt.figure()

play_task()
