"""
python3
決定論的ツリーバンディットタスク
二分木探索を参考に基準値を探る
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import Task
import agent


# 設定
NUM_ACTION = 4
LAYER = 1
START_STATE = 0

SIMULATION_TIMES = 100
EPISODE_TIMES = 5000
reference = 0.0
learning_rate = 0.1
discount_rate = 0.9
tau_alpha = 0.1
tau_gamma = 0.9
Ep = 0.0
discountEp = 0
ZETA = 0.01

NUM_AGENT = 1
task = Task.PTreeBandit(LAYER, SIMULATION_TIMES, EPISODE_TIMES)
NUM_STATE = task.get_num_state()
GOAL_STATE = task.get_goal_state()

player = agent.RS(learning_rate, discount_rate, reference,
                  tau_alpha, tau_gamma, NUM_STATE, NUM_ACTION,
                  "Q_learning")

print("バンディットの確率表示")
task.print_bandit()


def play_task():

    sumreward_for_graph = np.zeros((NUM_AGENT, EPISODE_TIMES))
    for n_simu in range(SIMULATION_TIMES):
        player.init_params()
        sumreward_for_graph = np.zeros(EPISODE_TIMES)

        print("Simu:{}".format(n_simu))
        for n_epi in range(EPISODE_TIMES):
            current_state = START_STATE
            sum_reward = 0

            while True:
                current_action = player.get_serect_action(current_state)

                task.evaluate_next_state(current_state, current_action)

                next_state = task.get_next_state()

                reward = task.evaluate_reward(current_state,
                                                current_action)

                sum_reward += reward
                player.update(current_state, next_state,
                              current_action, reward)

                current_state = task.get_next_state()

                # print("R:{}".format(player.get_reference()))
                # print("Q:{}".format(player.policy.get_Q()[0]))

                if current_state >= task.get_goal_state():
                    break

            sumreward_for_graph[n_epi] += sum_reward


    print("シミュレーション完了")
    print("バンディットの確率表示")
    task.print_bandit()

    print("獲得平均報酬グラフ表示")
    
    plt.plot((sumreward_for_graph) / SIMULATION_TIMES,
              label="RS")
    plt.legend()
    plt.title("reward")
    plt.xlabel("episode")
    plt.ylabel("reward")
    # plt.savefig("Sumreward_ave_time_development_Simu{}_Epi{}_Agent{}"
    #             .format(SimulationTimes, EpisodeTimes, n_agent))
    plt.show()
    # plt.figure()

play_task()
