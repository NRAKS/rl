"""
python3
簡易スイッチワールドで最短経路の獲得を目指す

Q学習エージェントを追加
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import Task
import Policy

#設定
n_Act = 4

BoardRow = 8
BoardCol = 10
n_state = BoardCol * BoardRow

StartState = 70
GoalState = (BoardCol * BoardRow) - 1

SimulationTimes = 5
EpisodeTimes = 500
#R_simple = 0
alpha = 0.1
gamma = 0.9
Talpha = 0.1
Tgamma = 0.9
Ep = 0.0
DicEp = 0
#タスク
Maze = Task.Criff_world(BoardRow, BoardCol, StartState, GoalState)
#エージェント
Q = Policy.Agent(alpha, gamma, n_Act, n_state, EpisodeTimes, SimulationTimes, Ep, DicEp)


def PlayTask():
    for n_Simu in range(SimulationTimes):
        Q.InitParameters()
        print("Simu:{}".format(n_Simu))
        for n_epi in range(EpisodeTimes):
            #print("Epi:{}".format(n_epi))
            CurrentStateQ = StartState
            
            #print("Epi:{}".format(n_epi))
            
            #Q学習
            while True:
                Q.SerectAction(CurrentStateQ)
                #print(f"serectact:{Q.GetSerectAction()}")
                Maze.EvaluateNextState(Q.GetSerectAction(),CurrentStateQ)
                NextStateQ = Maze.GetNextState()
                #報酬判定
                Reward = Maze.EvaluateReward(NextStateQ)
                
                #Q値のなどの更新
                Q.Update(Q.GetSerectAction(), CurrentStateQ, NextStateQ, Reward, n_epi)
                #崖に落ちた場合,スタート地点に移動することがあるので状態を再取得
                NextStateQ = Maze.GetNextState()
                #次状態に移る
                CurrentStateQ = NextStateQ
                
                if CurrentStateQ == GoalState:
                    Q.DicreaseEp()
                    #print("Q_agent Goal!")
                    break
                
    print("結果表示")
    #Reward平均表示
    
    # plt.plot(Q.culculationAve(), label = "Q learning ε 1.0 → 0 ({})".format(DicEp))
    # #plt.plot(Q.GetSumReward(),label = "Last Simu Q")
    # plt.legend()
    # plt.title("Output")
    # plt.xlabel("Episode")
    # plt.ylabel("Reward")
    # plt.show()
    # plt.savefig("EasyMaze1.png")

    #ステップ数表示
    #plt.plot(RS.GetSumStep()/SimulationTimes, label="RS simple, R = {}".format(R_simple))

    plt.plot(Q.GetSumStep() / SimulationTimes, label="Q ε 1.0 → 0(--{})".format(DicEp) )
    plt.legend()
    plt.title("step count")
    plt.xlabel("episode")
    plt.ylabel("step")
    plt.show()

PlayTask()