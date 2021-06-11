#import gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from random_process import OrnsteinUhlenbeckProcess

MEMORY_CAPACITY = 10000

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 3)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = F.softmax(self.fc3(out))
        #out = self.tanh(out)
        #ut = F.log_softmax(out,dim=1)
        return out


class Critic(nn.Module):
#class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=256 ,init_w = 3e-3):
        super(Critic, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w,init_w)
        self.linear3.bias.data.uniform_(-init_w,init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        

        return x


class Agent(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        s_dim = 2
        a_dim = 3
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.actor = Actor(s_dim, a_dim)
        self.actor_target = Actor(s_dim, a_dim)
        self.critic = Critic(s_dim, a_dim)
        self.critic_target = Critic(s_dim, a_dim)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr = self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr = self.critic_lr)
        self.buffer = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
    def policy_action(self, s0):
        s0 = torch.tensor(s0, dtype=torch.float).reshape((2,))
        a0 = self.actor(s0).detach().numpy() #+ np.random.uniform(-0.05,0.05,1)
        
        return a0
    
    def random_action(self):
        action = np.random.randint(0,3)
        return action
    
    
    def put(self, s0,a0,r1,s1): 
        transition = np.hstack((np.array(s0).reshape(-1),np.array(a0).reshape(-1),np.array(r1).reshape(-1),np.array(s1).reshape(-1)))
        index = self.pointer % MEMORY_CAPACITY
        self.buffer[index, :] = transition
        self.pointer += 1
    
    def replay(self):
        indices = np.random.choice(MEMORY_CAPACITY, size=self.batch_size)
        batch = self.buffer[indices,:]
        s0_batch = torch.FloatTensor(batch[:, :self.s_dim])
        a0_batch = torch.FloatTensor(batch[:,self.s_dim: self.s_dim + self.a_dim])
        r_batch = torch.FloatTensor(batch[:, -self.s_dim - 1: -self.s_dim])
        s1_batch = torch.FloatTensor(batch[:, -self.s_dim:])
        return s0_batch,a0_batch,r_batch,s1_batch
    
    def learn(self):
        if self.pointer<MEMORY_CAPACITY :
            return 
        
        _s0,_a0,_r1,_s1 = self.replay()
        
        def critic_learn():
            _a1 = self.actor_target(_s1).detach()
            y_true = _r1 + self.gamma * self.critic_target(_s1, _a1).detach()
            
            y_pred = self.critic(_s0, _a0)
            
            loss_fn = nn.MSELoss()
            loss = loss_fn(y_pred, y_true)
            #print("critic:",loss)
            self.critic_optim.zero_grad()
            loss.backward()
            self.critic_optim.step()
            
        def actor_learn():
            loss = -torch.mean( self.critic(_s0, self.actor(_s0)) )
            self.actor_optim.zero_grad()
            #print("actor:",loss)
            loss.backward()
            self.actor_optim.step()
        
                                        
        def soft_update(net_target, net, tau):
            for target_param, param  in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
        critic_learn()
        actor_learn()
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)

    

