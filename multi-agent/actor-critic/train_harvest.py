"""
複数エージェントでの合計報酬を見たい
合計報酬が基準を満たして居たら成功
それ以外は失敗
GRCエージェントに学習させる
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import Task
import agent
import sys

# 設定
NUM_ACTION = 5

BOARD_ROW = 7
BOARD_COL = 7
NUM_STATE = BOARD_COL * BOARD_ROW

START_STATE = 0
REWARD_STATE1 = (BOARD_COL * BOARD_ROW) - 1
REWARD_STATE2 = (BOARD_COL * BOARD_ROW) - 6

SIMULAITON_TIMES = 1
EPISODE_TIMES = 5000
reference = 0
learning_rate = 0.1
discount_rate = 0.9
tau_alpha = 0.1
tau_gamma = 0.9
eps_init = 1.0
eps_discount = 0.01
eps_min = 0.0
ZETA = 0.01

NUM_AGENT = 1

reference_g = 400 * NUM_AGENT



env = Task.HarvestWorld(BOARD_ROW, BOARD_COL, START_STATE, REWARD_STATE1, REWARD_STATE2)

player = [agent.GRC(learning_rate, discount_rate, reference_g, ZETA, tau_alpha, tau_gamma, NUM_STATE, NUM_ACTION, "Q_learning") for _ in range(NUM_AGENT)]

player_eps = [agent.eps_greedy(eps_init, eps_discount, eps_min, learning_rate, discount_rate, NUM_STATE, NUM_ACTION, policy="Q_learning") for _ in range(NUM_AGENT)]


def play_task():
    sumreward_for_graph_GRC = np.zeros((NUM_AGENT, EPISODE_TIMES))
    sumreward_for_graph_eps = np.zeros((NUM_AGENT, EPISODE_TIMES))

    for n_simu in range(SIMULAITON_TIMES):

        for i in range(len(player)):
            player[i].init_params()
            player_eps[i].init_params()
        
        # print("Simu:{}".format(n_simu))
        sys.stdout.write("\r%s/%s" % (str(n_simu), str(SIMULAITON_TIMES-1)))

        # GRCのトレーニング
        for n_epi in range(EPISODE_TIMES):
            # print("n_epi:{}".format(n_epi))
            sys.stdout.write("\r%s/%s" % (str(n_epi), str(EPISODE_TIMES-1)))
            sum_reward = np.zeros(len(player))
            current_state = np.full(NUM_AGENT, START_STATE)
            step = 0
            while True:
                for n_agent in range(NUM_AGENT):
                    current_action = player[n_agent].get_serect_action(tuple(current_state))
                    env.evaluate_next_state(current_action, current_state, n_agent)
                    next_state = env.get_next_state()

                    reward = env.evaluate_reward(next_state[n_agent])

                    sum_reward[n_agent] += reward

                    player[n_agent].update(tuple(current_state), tuple(next_state), current_action, reward)

                    current_state = next_state

                step += 1

                if step == 100:
                    break
            
            for n_agent in range(NUM_AGENT):
                sumreward_for_graph_GRC[n_agent, n_epi] += sum_reward[n_agent]
                print("rew:{}".format(np.sum(sum_reward)))
                player[n_agent].update_GRC_params(np.sum(sum_reward))
            # R_update = np.average(sum_reward)

        # eps-greedyのトレーニング
        # for n_epi in range(EPISODE_TIMES):
        #     sum_reward = np.zeros(len(player_eps))
        #     current_state = np.full(NUM_AGENT, START_STATE)
        #     step = 0
        #     while True:
        #         for n_agent in rnage(len(player_eps))
        #             current_action = player_eps[n_agent].get_serect_action(current_state)
        #             env.evaluate_next_state(current_action, current_state, n_agent)
        #             next_state = env.get_next_state()

        #             reward = env.evaluate_reward(next_state[n_agent])

        #             sum_reward[n_agent] += reward
        #             player_eps[n_agent].update(current_state[n_agent], next_state[n_agent], current_action, reward)

        #             current_state = next_state

        #             step += 1

        #             if step == 100:
        #                 break

        #         player_eps[n_agent].update_eps()
        #         print(sum_reward[n_agent])
        #         sumreward_for_graph_eps[n_agent, n_epi] += sum_reward[n_agent]
    print("シミュレーション完了")

    for n_agent in range(len(player)):
        plt.plot((sumreward_for_graph_GRC[n_agent]) / SIMULAITON_TIMES,
                 label="RS_{}".format(n_agent))
    plt.plot(sum(sumreward_for_graph_GRC) / SIMULAITON_TIMES, label="RSGRC_Group")

    for n_agent in range(len(player)):
        plt.plot((sumreward_for_graph_eps[n_agent]) / SIMULAITON_TIMES,
                 label="eps_{}".format(n_agent))
    plt.plot(sum(sumreward_for_graph_eps) / SIMULAITON_TIMES, label="eps_Group")

    plt.legend()
    plt.title("reward time development")
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.show()

play_task()


