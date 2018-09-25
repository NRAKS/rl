"""
actor-criticのテスト
グリッドワールドでの最適経路問題
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import Task
import model
import sys

# 設定
NUM_ACTION = 5

BOARD_ROW = 7
BOARD_COL = 7

START_STATE = 0
REWARD_STATE = (BOARD_COL * BOARD_ROW) - 1

SIMULAITON_TIMES = 1
EPISODE_TIMES = 5000
learning_rate = 0.1
discount_rate = 0.9


env = Task.GlidWorld(BOARD_ROW, BOARD_COL, START_STATE, REWARD_STATE)

player = model.actor_critic(num_state=env.get_num_state(), num_action=env.get_num_action())


def play_task():
    reward_graph = np.zeros(EPISODE_TIMES)
    step_graph = np.zeros(EPISODE_TIMES)

    for n_simu in range(SIMULAITON_TIMES):
        player.init_params()

        # sys.stdout.write("\r%s/%s" % (str(n_simu), str(SIMULAITON_TIMES-1)))

        # GRCのトレーニング
        for n_epi in range(EPISODE_TIMES):
            # print("n_epi:{}".format(n_epi))
            sys.stdout.write("\r%s/%s" % (str(n_epi), str(EPISODE_TIMES-1)))
            current_state = env.get_start_state()
            step = 0
            while True:
            
                current_action = player.serect_action(current_state)
                env.evaluate_next_state(current_action, current_state)
                next_state = env.get_next_state()

                reward = env.evaluate_reward(next_state)

                reward_graph[n_epi] += reward

                player.update(current_state, current_action, reward, next_state)

                current_state = next_state

                step += 1

                if step == 100 or next_state == env.get_goal_state():
                    step_graph[n_epi] += step
                    break
     
    print("シミュレーション完了")

    print("Q値表示")
    for n in range(env.get_num_state()):
        print("{}:{}".format(n, player.critic.get_Q()[n]))

    print("TD誤差表示")
    for n in range(env.get_num_state()):
        print("{}:{}".format(n, player.critic.get_Q()[n]))
    
    print("weight表示")
    for n in range(env.get_num_state()):
        print("{}:{}".format(n, player.actor.get_weight()[n]))

    print("グラフ表示")
    plt.plot(step_graph / SIMULAITON_TIMES, label="actor_critic")
    plt.legend()
    plt.title("step time development")
    plt.xlabel("episode")
    plt.ylabel("step")
    plt.show()

play_task()
