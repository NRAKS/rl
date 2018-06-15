#python3
"""
スイッチワールドをRS学習でこなすプログラム
未完成
"""
import random
import time
import numpy as np
import matplotlib.pyplot as plt

#action
NumOfAct = 4
Upper = 0
Lower = 1
Left = 2
Right = 3
def ActionClass(Action):
    if Action == 0:
        return "U"
    elif Action == 1:
        return "D"
    elif Action == 2:
        return "L"
    elif Action == 3:
        return "R"

#board
BoardRow = 7
BoardCol = 7
NumOfStates = (BoardRow * BoardCol)*3
StartState = 0
NumOfSwState = 3
SwState = {0:10, 1:89, 2:134}
FlagSw = {}
MinStepCnt = BoardCol + 1

#const
Alpha = 0.1
Gamma = 0.9
Ep = 0.1
R = 6
Z = 0.05

#preference
NumOfSimulation = 10
NumOfEpisode = 150

#macro
def CoordToState(row, col, FlagCnt):
    return (((row) * (BoardCol)) + (col)) + (FlagCnt * 49)
def StateToRow(State, FlagCnt):
    return ((int)((State-(FlagCnt*49)) / BoardCol))
def StateToCol(State, FlagCnt):
    return ((State-(49*FlagCnt)) % BoardCol)

#gloval variables
ValueQ = {}
T = {}
Tcurrent = {}
Tpost = {}
r = {}
SimulationTimes = 0
RoopTimes = 0
StepCnt = 0
SumReward = {}
SumRewardForAve = {}
SumStep = {}

#引数の状態において引数の行動を行なった場合に遷移する先の状態を返す
def EvaluateNextState(action, state, FlagCnt):
    row = StateToRow(state,FlagCnt)
    col = StateToCol(state,FlagCnt)

    if action == Upper:
        if (row - 1) > 0:
            row-=1
    elif action == Lower:
        if (row + 1) < (BoardRow-1):
            row+=1
    elif action == Right:
        if (col + 1) < (BoardCol-1):
            col+=1
    elif action == Left:
        if (col - 1) > 0:
            col-=1

    #座標を状態に変換して返す
    return CoordToState(row,col,FlagCnt)

#引数の状態から得られる報酬を返す
def EvaluateReward(FlagCnt):
    #スイッチを全て踏んだ場合
    if FlagCnt == 3:
        FlagCnt == 0
        #報酬１を返す
        return 1
    else:
        return 0

#引数の状態においてグリーディ上策に従う場合、次にとるべき行動を返す
def GreedyPolicy(state):
    m = 0
    NumOfMax = 0
    MaxAction = {}

    for act in range(NumOfAct):
        #各行動毎のQ値を求める
        TempQ = ValueQ[state, act]

        if TempQ > m:
            #最大値を更新する
            m = TempQ
            #新たな最大値を取る行動の個数を記憶する
            NumOfMax = 1
            #新たな最大値を取る行動を記憶する
            MaxAction[0] = act

        elif TempQ == m:
            #この最大値を取る行動の個数を更新する
            NumOfMax+=1
            #最大値を取る行動を追加する
            MaxAction[NumOfMax - 1] = act
    
    if NumOfMax > 1:
        SelectedAction = MaxAction[random.randrange(1000) % NumOfMax]
    else:
        SelectedAction = MaxAction[0]
    
    return SelectedAction

def RandomAction():
    return random.randrange(1000) % 4

def EpsilonGreedyPolicy(state):
    if random.random() > Ep:
        return GreedyPolicy(state)
    else:
        return RandomAction()

#RSに基づいて行動を返す
def RS(state,ValueQ,NumOfAct,T,R,DG):
    """ToDo ここでR[si]を計算する"""
    MaxValueQ = max(ValueQ[state,x] for x in range(NumOfAct))
    r[state] = MaxValueQ-Z*DG
    rs = {}
    for act in range(NumOfAct):
        rs[state,act] = T[state,act] * (ValueQ[state,act] - r[state])
    MaxOfRS = max(rs[state,x] for x in range(NumOfAct))
    NRS = np.asarray([rs[state,0],rs[state,1],rs[state,2],rs[state,3]])
    idx = np.where(NRS == MaxOfRS)
    SerectedAct = random.choice(idx[0])
        
    return SerectedAct


#GRCの計算
def CalculationR(state,ValueQ):
    MaxValueQ = max(ValueQ[state,x] for x in range(NumOfAct))
    Etmp = Reward/Step
    EG = (Etmp + Gamma*(NG*EG))/(1+Gamma*NG)
    DG = min(EG-RG,0)
    return MaxValueQ - Z * DG
    

#シミュレーションごとのステップ数と報酬を初期化
for i in range(NumOfEpisode):
    SumStep[i] = 0
    SumRewardForAve[i] = 0
SumReward = np.asarray([])

StartTime = time.time()

def InitializationFlagSw(FlagSw):
    #スイッチの数分ループさせる
    for i in range(NumOfSwState):
        #ステップが１の時(スタート時)
        if StepCnt == 0:
            FlagSw[i] = 0

#各シミュレーションに対して繰り返す
for SimulationTimes in range(NumOfSimulation):
    for i in range(NumOfStates):
        for j in range(NumOfAct):
            ValueQ[i, j] = 0
            T[i, j] = 0
            Tcurrent[i, j] = 0
            Tpost[i, j] = 0
            r[i] = 0
    EG = 0
    NG = 0
    print("SimulationCnt: "+str(SimulationTimes))
    for RoopTimes in range(NumOfEpisode):
        #状態の初期化
        CurrentState = StartState
        #歩数の初期化
        StepCnt = 0
        #SumStep[RoopTimes] = 0
        #１エピソードあたりの報酬の合計値を初期化
        #SumReward[RoopTimes] = 0
        SumRewardsPerEpisode = 0
        #スイッチ周りの設定
        FlagCnt = 0
        InitializationFlagSw(FlagSw)
        if Ep > 0:
            Ep -= 0.005
        while True:
            DG = min([EG-R,0])
            #方策に従って、Sでの行動aを選択
            #if(RoopTimes<990):
            if RoopTimes > 90:
                CurrentAction = GreedyPolicy(CurrentState)
            else:    
                CurrentAction = RS(CurrentState,ValueQ,NumOfAct,T,R,DG)
            #else:
            #    CurrentAction = GreedyPolicy(CurrentState)

            #行動aを取り、r, s'を観測する
            NextState = EvaluateNextState(CurrentAction, CurrentState,FlagCnt)
            #フラグ判定
            if NextState == SwState[FlagCnt]:
                #print("スイッチ踏んだ")
                #print("step:{}".format(StepCnt))
                FlagSw[FlagCnt] = 1
                FlagCnt+=1
                NextState += 49
            
            Reward = EvaluateReward(FlagCnt)

            if Reward == 1:
                for i in range(len(SwState)):
                    FlagSw[i] = 0
                FlagCnt = 0
                NextState -= 147
            #エピソードあたりの報酬の合計値を更新
            SumRewardsPerEpisode += Reward

            #次の状態での行動を選択する
            #NextAction = RS(NextState,ValueQ,NumOfAct,T,R,DG)
            #現在の状態に対する各行動でのQ値の最大値を計算する
            
            NextStateValueQ = np.asarray([ValueQ[NextState,0],ValueQ[NextState,1],ValueQ[NextState,2],ValueQ[NextState,3]])
            MaxValueQ = max(ValueQ[NextState,x] for x in range(NumOfAct))
            idx = np.where(NextStateValueQ == MaxValueQ)
            ActionUP = random.choice(idx[0])
           
            #Q(s,a)を更新
            ValueQ[CurrentState, CurrentAction] += Alpha * (Reward + Gamma * MaxValueQ - ValueQ[CurrentState, CurrentAction])

            #τ値を更新
            Tcurrent[CurrentState,CurrentAction] += 1

            Tpost[CurrentState,CurrentAction] += Alpha*(Gamma*T[NextState,ActionUP]-Tpost[CurrentState,CurrentAction])

            T[CurrentState,CurrentAction] = Tcurrent[CurrentState,CurrentAction]+Tpost[CurrentState,CurrentAction]

            #状態をsからS’に遷移する
            CurrentState = NextState
            #歩数を増加させる
            StepCnt += 1
            #終了推移を行う
            #ステップ数が１００に達した時
            if  StepCnt == 100:
                print("EpisodeCnt:{:5}".format(RoopTimes)+",Reward:{:2}".format(SumRewardsPerEpisode))
                break
        #全シミュレーションを通して、エピソード毎の報酬の総和を求める
        #SumReward = np.append(SumReward,[SumRewardsPerEpisode])
        SumRewardForAve[RoopTimes] += SumRewardsPerEpisode
        SumStep[RoopTimes] += StepCnt
        
        """ToDo ここでEtmp,EG,NGを更新する"""
        Etmp = SumRewardsPerEpisode
        EG = (Etmp + Gamma * (NG * EG)) / (1 + Gamma * NG)
        NG = 1 + Gamma * NG

        #print("Episode: "+ str(RoopTimes))
        #ステップ数の表示
        #print("stepCnt: " + str(StepCnt))
        """
        # Q 値の表示
        for State in range(48):
            for Action in range(4):
                print("State:" + str(State) + " Action:" + ActionClass(Action)+ "   ValueQ: "+str(ValueQ[State, Action]))
        """
for i in range(NumOfEpisode):
    SumReward = np.append(SumReward,[SumRewardForAve[i]/NumOfSimulation])
x = np.array(range(NumOfEpisode))
    
plt.plot(x,SumReward)
plt.show()
for i in range(NumOfAct):
    print("状態137,行動{}".format(i)+" Q値{:5}".format(ValueQ[137,i])+" T値{:5}".format(T[137,i]))
EndTime = time.time()


#全シミュレーションを通して、エピソード毎のステップ数の平均を出力する
"""
for i in range(NumOfEpisode):
    #AveOfStep = SumStep[i] / NumOfSimulation
    print("episode: {:4}".format(i)+" Avg of Reward: {}".format(SumReward[i
    ]/NumOfSimulation))
"""
#実行時間の表示
print("計算時間は{:.4f}秒でした。" .format(EndTime - StartTime))
