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

# シミュレーション設定
SIMULATION_TIMES = 1000
EPISODE_TIMES = 5000
# タスク環境設定
LAYER = 5
task = Task.DTreeBandit(LAYER, SIMULATION_TIMES, EPISODE_TIMES)
print("バンディット確率表示")
task.print_bandit()
START_STATE = 0
GOAL_STATE = task.get_goal_state()
NUM_ACTION = 4
NUM_STATE = task.get_num_state()
# エージェント設定(設定した数字の-1で実行される。)
NUM_AGENT = 6
reference = 0
# R_simple = 0
learning_rate = 0.1
discount_rate = 1.0
T_alpha = 0.1
T_gamma = 0.9
Ep = 1.0
discount_Ep = 0.005


def sum_ideal_reward_epi(ideal_reward, num_episode):
    SumRewardperEpi = np.zeros((num_episode))
    for i in range(num_episode):
        if i >= 1:
            SumRewardperEpi[i] = ideal_reward + SumRewardperEpi[i-1]
        else:
            SumRewardperEpi[i] = ideal_reward
    return SumRewardperEpi


def culculation_reference_share(Q1, Q2):
    maxQ = np.zeros((NUM_STATE))
    for state in range(NUM_STATE):
        maxQ[state] = max(max(Q1[state]), max(Q2[state]))
    return maxQ


def play_task():
    average_reward_list = np.zeros((NUM_AGENT, EPISODE_TIMES))
    regret_list = np.zeros((NUM_AGENT, EPISODE_TIMES))
    reference_share_list = np.zeros((NUM_AGENT, EPISODE_TIMES))

    for n_agent in range(1, NUM_AGENT):
        print("Agent num:{}".format(NUM_AGENT))
        agent = SimulationManager(NUM_AGENT, SIMULATION_TIMES,
                                  EPISODE_TIMES, task)
        agent.addRS(reference, NUM_ACTION, NUM_STATE, learning_rate,
                    discount_rate, T_alpha, T_gamma, EPISODE_TIMES,
                    SIMULATION_TIMES)

        ideal_reward = task.seek_ideal()
        print("最適報酬:{}".format(ideal_reward))
        sum_ideal_reward = sum_ideal_reward_epi(ideal_reward, EPISODE_TIMES)
        print("最適報酬和：{}".format(sum_ideal_reward))

        for n_simu in range(SIMULATION_TIMES):
            agent.InitParameter()
            print("Simu:{}".format(n_simu))
            for n_epi in range(EPISODE_TIMES):
                # print("Epi:{}".format(n_epi))
                agent.play(n_epi)

        average_reward_list[n_agent] = agent.PlotAverageReward()
        regret_list[n_agent] = agent.PlotRegret(sum_ideal_reward)
        reference_share_list[n_agent] = agent.GetR_share() / SIMULATION_TIMES

    with open(f"R_Sharelist_{LAYER}L_{SIMULATION_TIMES}S_{EPISODE_TIMES}epi.pkl",
              mode="wb") as f:
        pkl.dump(reference_share_list, f)

    print("平均報酬獲得の結果表示")
    Ideal = np.full(EPISODE_TIMES, ideal_reward)
    for n in range(1, NUM_AGENT):
        plt.plot(average_reward_list[n], label="Agent num:{}" .format(n))
    plt.plot(Ideal, label="Optimum")
    plt.legend()
    # plt.title("output")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig("TreeBandit_RewardAve_{}L{}s{}e.png"
                .format(LAYER, SIMULATION_TIMES, EPISODE_TIMES))
    # plt.show()
    plt.figure()

    print("regret表示")
    for n in range(1, NUM_AGENT):
        # print("エージェント数:{}のグラフ出力".format(n))
        plt.plot(regret_list[n], label="Agent num:{}" .format(n))
    plt.legend()
    # plt.title("regret output")
    plt.xlabel("Episode")
    plt.ylabel("regret")
    plt.savefig("TreeBandit_regret_{}L{}s{}e.png"
                .format(LAYER, SIMULATION_TIMES, EPISODE_TIMES))
    plt.figure()
    # plt.show()


play_task()
