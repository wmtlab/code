# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 13:03:38 2021


"""


import env
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from DDPG import Agent
import matplotlib.pyplot as plt
import gym
import numpy as np

MEMORY_CAPACITY = 3000
env = env.Env()
#problem = "Pendulum-v0"
#env = gym.make(problem)
#env.reset()
#env.render()

params = {
    'env': env,
    'gamma': 0.5, 
    'actor_lr': 0.0001, 
    'critic_lr': 0.0001,
    'tau': 0.02,
    'capacity': 10000, 
    'batch_size': 32,
    }

agent = Agent(**params)
reward = []
step = 0
for episode in range(500):
    s0 = env.reset()
    #s_origin = s0[0]
    t = []
    y = []
    p = []
    t0 = 0
    episode_reward = 0
    
    for i in range(500):
        #env.render()
        #ratio = s_origin/s0
        

        a0 = agent.policy_action(s0)
            
            #a0 = a0*ratio
        #a0 = agent.act(s0)
        s1, r1, done, _ = env.step(a0.argmax())
        # print("send_rate:%d"%s1[0])
        # print("loss package:%d"%s1[1])
        # print("reward:%d"%r1)
        agent.put(s0, a0, r1, s1)
        #t.append(t0+env.dt)
        #t0 = t0+env.dt
        #y.append(s1[0])
        episode_reward += r1 
        s0 = s1
        #p.append(env.Force)
        #print(step)
        agent.learn()
        if(done == True):
            break
    reward.append(episode_reward)

    print(episode, ': ', episode_reward)


