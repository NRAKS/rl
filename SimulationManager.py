#シミュレーション関連の機能まとめ

import Policy
import TaskMultiAgent
import numpy as np
import matplotlib.pyplot as plt
import csv

class SimuManager():
    def __init__(self, n_agent, simulationtimes, episodetimes, task):
        self.N = n_agent
        self.AgentList = None
        self.task = task
        self.simulationtimes = simulationtimes
        self.episodetimes = episodetimes
        self.R_share_recode = np.zeros((episodetimes))
        self.RSflag = 0

    #Qエージェントの追加(list管理)
    def addQagent(self, alpha, gamma, n_act, n_state, n_epi, n_simu, Ep, DicEp):
        self.AgentList = [Policy.Agent(alpha, gamma, n_act, n_state, n_epi, n_simu, Ep, DicEp) for _ in range(self.N)]

    def GetAgent(self):
        return self.AgentList

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
    