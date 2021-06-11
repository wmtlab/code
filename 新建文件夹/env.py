# -*- coding: utf-8 -*-
"""
Created on Sun May  9 14:24:28 2021

@author: Teng Ma
"""
import numpy as np
import random

class Env():
    def __init__(self):
        self.state = None
        self.action = None
        self.reward = None
        self.s_threshold = 40
        return 
    
    def reset(self):
        send_rate = np.random.random()*self.s_threshold
        self.state = np.array([send_rate,0])  #state[0]是send_rate,state[1]是是否丢包，0未丢包，1丢包
        self.action = 1
        return self.state 
        
    def step(self,action):
        self.action = action
        event_info = np.random.randint(0,2) # 是否丢包事件，后期可以替换为仿真函数
        if(event_info == 0):
            self.reward = self.state[0]
        else:
            self.reward = -self.state[0]*20
        if(self.action==0):
            send_rate = self.state[0] - 10
        elif(self.action == 1):
            send_rate = self.state[0]
        else:
            send_rate = self.state[0] + 10
        send_rate = np.clip(send_rate,0,self.s_threshold)
        self.state = np.array([send_rate,event_info])
        return self.state,self.reward,0,{}

        
        