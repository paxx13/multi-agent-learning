import sys
import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, num_outputs):
        super(Actor, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        torch.nn.init.xavier_normal_(self.linear1.weight, 0.1)
        self.linear1.bias.data.mul_(0.1)
        self.ln1 = nn.LayerNorm(hidden_size)

        self.linear2 = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_normal_(self.linear2.weight, 0.1)
        self.linear2.bias.data.mul_(0.1)
        self.ln2 = nn.LayerNorm(hidden_size)

        self.mu = nn.Linear(hidden_size, num_outputs)
        torch.nn.init.xavier_normal_(self.mu.weight, 0.1)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.1)

    def forward(self, inputs):
        x = inputs
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)
        mu = torch.tanh(self.mu(x))
        return mu

        
class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, num_outputs):
        super(Critic, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        torch.nn.init.xavier_normal_(self.linear1.weight, 0.1)
        self.linear1.bias.data.mul_(0.1)
        self.ln1 = nn.LayerNorm(hidden_size)

        self.linear2 = nn.Linear(hidden_size+num_outputs, hidden_size)
        torch.nn.init.xavier_normal_(self.linear2.weight, 0.1)
        self.linear2.bias.data.mul_(0.1)
        self.ln2 = nn.LayerNorm(hidden_size)        

        self.V = nn.Linear(hidden_size, 1)
        torch.nn.init.xavier_normal_(self.V.weight, 0.1)
        self.V.bias.data.mul_(0.1)

    def forward(self, inputs, actions):
        x = inputs
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)

        x = torch.cat((x, actions), 1)
        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)
        V = self.V(x)
        return torch.tanh(V)

        
class Agent(object):
    def __init__(self, action_bound, gamma, tau, hidden_size, num_inputs, num_outputs):

        self.num_inputs = num_inputs
        self.action_bound = action_bound

        self.actor = Actor(hidden_size, self.num_inputs, num_outputs)
        self.actor_target = Actor(hidden_size, self.num_inputs, num_outputs)
        self.actor_optim = Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(hidden_size, self.num_inputs, num_outputs)
        self.critic_target = Critic(hidden_size, self.num_inputs, num_outputs)
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-3)

        self.gamma = gamma
        self.tau = tau

        hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)


    def select_action(self, state, action_noise=None):
        self.actor.eval()
        mu = self.actor((Variable(torch.Tensor(state))))
            
        self.actor.train()
        mu = mu.data

        if action_noise is not None:
            mu += torch.Tensor(action_noise.noise())

        return torch.autograd.Variable(mu, requires_grad=False)* self.action_bound


    def train(self, batch):
        state_batch = Variable(torch.FloatTensor(batch.state))
        action_batch = Variable(torch.FloatTensor(batch.action))
        # normalize rewards
        rewards = (batch.reward - np.array(batch.reward).mean()) / np.array(batch.reward).std()
        reward_batch = Variable(torch.FloatTensor(batch.reward))
        mask_batch = Variable(torch.FloatTensor(batch.mask))
        next_state_batch = Variable(torch.FloatTensor(batch.next_state))
        
        next_action_batch = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(next_state_batch, next_action_batch)

        reward_batch = reward_batch.unsqueeze(1)        
        
        mask_batch = mask_batch.unsqueeze(1)
        expected_state_action_batch = reward_batch + (self.gamma * mask_batch * next_state_action_values)

        self.critic_optim.zero_grad()

        state_action_batch = self.critic((state_batch), (action_batch))

        value_loss = F.mse_loss(state_action_batch, expected_state_action_batch)
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.1)
        self.critic_optim.step()

        self.actor_optim.zero_grad()

        policy_loss = -self.critic((state_batch),self.actor((state_batch)))

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.1)
        self.actor_optim.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()

    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/ddpg_actor_{}_{}".format(env_name, suffix) 
        if critic_path is None:
            critic_path = "models/ddpg_critic_{}_{}".format(env_name, suffix) 
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
        if critic_path is not None: 
            self.critic.load_state_dict(torch.load(critic_path))