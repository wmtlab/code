from simple_emulator import CongestionControl
from simple_emulator import BlockSelection
from simple_emulator import constant
from simple_emulator import objects
from objects.sender import Sender
from objects.packet import Packet
import numpy as np;
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from random_process import OrnsteinUhlenbeckProcess
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt
import numpy as np
import random

EVENT_TYPE_FINISHED='F'
EVENT_TYPE_DROP='D'
EVENT_TYPE_TEMP='T'

# Superparameters
import numpy as np
import random
EPISODE = 20
########################################  DDPG  ################################

MEMORY_CAPACITY = 3000

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=40, hidden2=30, init_w=3e-3):
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
    def __init__(self, num_inputs, num_actions, hidden_size=30 ,init_w = 3e-3):
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
                                                                                    #以上AC经典模式

class Agent(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        s_dim = 2                                                           #状态维度2
        a_dim = 3                                                           #动作维度3
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.actor = Actor(s_dim, a_dim)                        #actor网络
        self.actor_target = Actor(s_dim, a_dim)            #actor目标网络
        self.critic = Critic(s_dim, a_dim)                         #critic网络
        self.critic_target = Critic(s_dim, a_dim)             #critic目标网络
        self.actor_optim = optim.Adam(self.actor.parameters(), lr = self.actor_lr)     #优化器，待优化参数，学习率
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
    
    def replay(self):                                                     #replay存储
        indices = np.random.choice(MEMORY_CAPACITY, size=self.batch_size)
        batch = self.buffer[indices,:]
        s0_batch = torch.FloatTensor(batch[:, :self.s_dim])
        a0_batch = torch.FloatTensor(batch[:,self.s_dim: self.s_dim + self.a_dim])
        r_batch = torch.FloatTensor(batch[:, -self.s_dim - 1: -self.s_dim])
        s1_batch = torch.FloatTensor(batch[:, -self.s_dim:])
        return s0_batch,a0_batch,r_batch,s1_batch
    
    def learn(self):                                                       #critic和actor的learn
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
        
                                        
        def soft_update(net_target, net, tau):                   #软更新
            for target_param, param  in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
        critic_learn()
        actor_learn()
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)


##############################################  train  #####################################
class RL(CongestionControl):

    def __init__(self):
        super(RL, self).__init__()
        self.USE_CWND=False
        self.send_rate = 40.0
        self.cwnd = 5000

        self.counter = 0 # EPISODE counter

        self.result_list = []
        self.last_state = []
        self.last_state = np.array([40,0])

    def estimate_bandwidth(self, cur_time, event_info):     #定义函数用来控制发送时的传送速率
        event_type = event_info["event_type"]                    #将事件信息中的类型（是否丢包等）
        event_time = cur_time                                            #将当前时间给到
        params = {                                                                          #各个参数集中地
            'gamma': 0.9, 
            'actor_lr': 0.001, 
            'critic_lr': 0.001,
            'tau': 0.02,
            'capacity': 10000, 
            'batch_size': 32,
            }
        agent = Agent(**params)
        reward = []
        step = 0

        # Preparing for the bandwidth estimator's decision
        if event_type == EVENT_TYPE_DROP:                      #如果丢包了，将1写进result_list中，并对counter++
            lost = 1
        else:
            lost = 0

        self.counter += 1
        if self.counter == EPISODE:                     # 进来20个包之后？开始选择动作？
            self.counter = 0                                  #将counter清零重新计数新的20个
            #print()
            #print("EPISODE:")
            #print()

            # reward
            r1 = 0                                                  
            for i in range(EPISODE):
                if lost== 0:
                    r1 += self.send_rate
                else:
                    r1 += -self.send_rate * EPISODE

            # current_state
            s1=np.array([self.send_rate,lost])

            a0 = agent.policy_action(s1)

            if a0.any() == 0:                                               
                self.send_rate += 10.0
            elif a0.any() == 1:
                self.send_rate += 0.0
            else:
                self.send_rate += -10.0
                if self.send_rate < 40.0:
                    self.send_rate = 40.0

            # last state                                             
            s0 = self.last_state
            agent.put(s0, a0, r1, s1)
            self.last_state = s1
            agent.learn()



# Your solution should include packet selection and bandwidth estimator.
# We recommend you to achieve it by inherit the objects we provided and overwritten necessary method.
class MySolution(BlockSelection, RL):

    def select_block(self, cur_time, block_queue):
        '''
        The alogrithm to select the block which will be sended in next.
        The following example is selecting block by the create time firstly, and radio of rest life time to deadline secondly.
        :param cur_time: float
        :param block_queue: the list of Block.You can get more detail about Block in objects/block.py
        :return: int
        '''
        def is_better(block):
            best_block_create_time = best_block.block_info["Create_time"]
            cur_block_create_time = block.block_info["Create_time"]
            # if block is miss ddl
            if (cur_time - cur_block_create_time) >= block.block_info["Deadline"]:
                return False
            if (cur_time - best_block_create_time) >= best_block.block_info["Deadline"]:
                return True
            if best_block_create_time != cur_block_create_time:
                return best_block_create_time > cur_block_create_time
            return (cur_time - best_block_create_time) * best_block.block_info["Deadline"] > \
                   (cur_time - cur_block_create_time) * block.block_info["Deadline"]

        best_block_idx = -1
        best_block= None
        for idx, item in enumerate(block_queue):
            if best_block is None or is_better(item) :
                best_block_idx = idx
                best_block = item

        return best_block_idx

    def on_packet_sent(self, cur_time):
        """
        The part of solution to update the states of the algorithm when sender need to send pacekt.
        """
        return super().on_packet_sent(cur_time)

    def cc_trigger(self, cur_time, event_info):
        """
        The part of algorithm to make congestion control, which will be call when sender get an event about acknowledge or lost from reciever.
        See more at https://github.com/AItransCompetition/simple_emulator/tree/master#congestion_control_algorithmpy.
        """
        # estimate the bandwidth
        super().estimate_bandwidth(cur_time, event_info)
        
        # set cwnd or sending rate in sender
        return {
            "cwnd" : self.cwnd,
            "send_rate" : self.send_rate,
        }