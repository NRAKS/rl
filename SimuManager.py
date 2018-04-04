#シミュレーション関連の機能まとめ

import Policy
from Task import TreeBandit
import numpy as np
import matplotlib.pyplot as plt
import csv

class SimulationManager():
    def __init__(self, n_agent, simulationtimes, episodetimes, task):
        self.N = n_agent
        self.AgentList = None
        self.task = task
        self.simulationtimes = simulationtimes
        self.episodetimes = episodetimes
        self.R_share_recode = np.zeros((episodetimes))
        self.RSflag = 0

    #RSエージェントの追加(list管理)
    def addRS(self, R, N_act, N_state, Alpha, Gamma, TAlpha, TGamma, N_Epi, N_Simu):
        self.AgentList = [Policy.RS(R, N_act, N_state, Alpha, Gamma, TAlpha, TGamma, N_Epi, N_Simu) for _ in range(self.N)]
        self.RSflag = 1

    #Qエージェントの追加(list管理)
    def addQagent(self, alpha, gamma, n_act, n_state, n_epi, n_simu, Ep, DicEp):
        self.AgentList = [Policy.Agent(alpha, gamma, n_act, n_state, n_epi, n_simu, Ep, DicEp) for _ in range(self.N)]

    def GetAgent(self):
        return self.AgentList

    def CulculationRshare(self):
        maxQ = np.zeros((self.task.GetNumState()))
        Qlist = np.array([self.AgentList[n].GetQ() for n in range(len(self.AgentList))])
        #print("{}".format(Qlist))
        maxQ = Qlist.max(axis = 2).max(axis = 0)
        #print("{}".format(maxQ))
        # for state in range(self.task.GetNumState()):
        #     for n in range(len(self.AgentList)):
        # for state in range(self.task.GetNumState()):
        #     maxQ[state] = max(Qlist[state])
        return maxQ

    def InitParameter(self):
        for n in range(len(self.AgentList)):
            self.AgentList[n].InitParameters()

    def play(self, n_epi):
        for n in range(len(self.AgentList)):
            self.AgentList[n].InitState(self.task.GetStartstate())
        for n in range(len(self.AgentList)):
            while True:
                #print("{}".format(n))
                CurrentState = self.AgentList[n].GetCurrentstate()
                self.AgentList[n].SerectAction(CurrentState)
                self.task.EvaluateNextState(CurrentState,self.AgentList[n].GetSerectAction())
                NextState = self.task.GetNextState()
                #print("Next:{}".format(NextStateUCB1))
                Reward = self.task.EvaluateReward(CurrentState,self.AgentList[n].GetSerectAction())
                #print("reward:{}".format(Reward))
                self.AgentList[n].Update(self.AgentList[n].GetSerectAction(), CurrentState, NextState, Reward, n_epi)
                self.AgentList[n].Updatestate(NextState)
                if NextState >= self.task.GetGoalState():
                    break
        
        if self.RSflag == 1:
            R_share = self.CulculationRshare()
            self.R_share_recode[n_epi] += R_share[0]
            for n in range(len(self.AgentList)):
                self.AgentList[n].ShareReference(R_share)
                self.AgentList[n].CountRsub(n_epi)
    
    def GetR_share(self):
        return self.R_share_recode

    def PlotAverageReward(self):
    
        SumReward = np.zeros((self.episodetimes))

        for n in range(len(self.AgentList)):
            #plt.plot(self.AgentList[n].culculationAve(),label = "RS{}".format(n))
            SumReward += self.AgentList[n].culculationAve()
        
        Ave = SumReward / len(self.AgentList)
        return Ave
        # plt.plot(Ave, label="RSpair")
        # plt.legend()
        # plt.title("Output")
        # plt.xlabel("Episode")
        # plt.ylabel("Reward")
        # plt.show()
        # with open('TreeBandit_AveReward.csv', 'a', encoding='UTF-8') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(Ave)

    def PlotRegret(self, SumRewardIdeal):
    
        sumregret = np.zeros((self.episodetimes))
        for n in range(len(self.AgentList)):
            #print("sumrewardideal:{}".format(SumRewardIdeal))
            #print("Agent{} sumreward:{}".format(n, self.AgentList[n].GetSumReward()))
            regret = SumRewardIdeal - self.AgentList[n].GetSumReward()
            #plt.plot(regret, label="RS{}".format(n))
            sumregret += regret
        RegretAve = sumregret / len(self.AgentList)
        return RegretAve
        # plt.plot(sumregret/len(self.AgentList), label = "Average")
        # # for n in range(len(self.AgentList)):
        # #     SumReward += self.AgentList[n].GetSumReward()
        # # plt.plot(SumRewardIdeal - (SumReward / len(self.AgentList)), label = "RSpair")
        # plt.legend()
        # plt.title("regret output")
        # plt.xlabel("Episode")
        # plt.ylabel("regret")
        # plt.show()

        