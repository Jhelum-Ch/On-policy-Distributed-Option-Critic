import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#Hyperparameters
learning_rate = 0.01
gamma = 0.99

class Broadcast_net(nn.Module):
    def __init__(self):
        super(Broadcast_net, self).__init__()
        self.agent_state_space = len(env.cell_list)
        #self.joint_observation = joint_obs
        
        
        self.l1 = nn.Linear(self.agent_state_space, 128, bias=False)
        self.l2 = nn.Linear(128, 2, bias=False) #output is binary
        
        self.gamma = gamma
        
        # Episode policy and reward history 
        self.boradcast_history = Variable(torch.Tensor()) 
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []
        
    def forward(self, x):
        model = torch.nn.Sequential(self.l1,nn.Dropout(p=0.5),nn.ReLU(),self.l2,nn.Softmax(dim=-1))
        return model(x)
    
#joint_obs = [(21,0),None,(18,-1)]
broadcast = Broadcast_net()
optimizer = torch.optim.Adam(broadcast.parameters(), lr=learning_rate)

def select_broadcast(state):
    state = torch.from_numpy(state).type(torch.FloatTensor)
    state = broadcast(Variable(state))
    c = Categorical(state)
    broadcast_action = c.sample()
    
    # Add log probability of our chosen action to our history    
    if broadcast.broadcast_history.dim() != 0:
        broadcast.broadcast_history = torch.cat([broadcast.broadcast_history, c.log_prob(broadcast_action)])
    else:
        broadcast.broadcast_history = (c.log_prob(broadcast_action))
    return broadcast_action

def update_broadcast_policy(broadcast):
    R = 0
    rewards_list = []
    
    # Discount future rewards back to the present using gamma
    for r in broadcast.reward_episode[::-1]:
        R = r + broadcast.gamma * R
        rewards.insert(0,R)
        
    # Scale rewards
    rewards_list = torch.FloatTensor(rewards_list)
    rewards_list = (rewards_list - rewards_list.mean()) / (rewards_list.std() + np.finfo(np.float32).eps)
    
    # Calculate loss for policy gradient update
    loss = (torch.sum(torch.mul(broadcast.broadcast_history, Variable(rewards_list)).mul(-1), -1))
    
    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    #Save and intialize episode history counters
    broadcast.loss_history.append(loss.data[0])
    broadcast.reward_history.append(np.sum(broadcast.reward_episode))
    broadcast.broadcast_history = Variable(torch.Tensor())
    broadcast.reward_episode= []


# def train_agent(agent, agent_belief_pmf, episodes):
#     #running_reward = 10
#     for episode in range(episodes):
        
        
#         agent_state = env.reset()[agent] # Reset environment and record the starting state
#         done = False 
        
#         for iteration in range(1000):
            
#             # TODO: augment belief
            
            
#             #sample agent_state according to agent's belief pmf
#             sample_agent_state = np.random.choice(env.cell_list,p=agent_belief_pmf)
            
#             # TODO: augment action from pi-policy net
#             # take action according to 'pi_policy_net'
#             agent_action = select_action(agent_state)
#             agent_broadcast_action = select_broadcast(agent_state)
#             # Step through environment using chosen action
#             agent_next_state, reward, done, _ = env.step(agent_action.data[0])
#             # modify the following
#             reward += env.broadcast_penalty*agent_broadcast_action + np.linalg.norm(sample_agent_state, agent_state)*env.selfishness_penalty*(1-agent_broadcast_action)
            
#             # Save reward
#             broadcast.reward_episode.append(reward)
#             if done:
#                 break
                
# #         # Used to determine when the environment is solved.
# #         running_reward = (running_reward * 0.99) + (time * 0.01)
            
#         update_broadcast_policy()