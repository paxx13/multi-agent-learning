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
    def __init__(self, hidden_size, num_inputs, num_actions):
        super(Critic, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        torch.nn.init.xavier_normal_(self.linear1.weight, 0.1)
        self.linear1.bias.data.mul_(0.1)
        self.ln1 = nn.LayerNorm(hidden_size)

        self.linear2 = nn.Linear(hidden_size+num_actions, hidden_size)
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
        return V


class Agent(object):
    def __init__(self, gamma, tau, hidden_size, num_inputs, num_outputs, critic_in_size, critic_act_size):

        self.num_inputs = num_inputs

        self.actor = Actor(hidden_size, self.num_inputs, num_outputs)
        self.actor_target = Actor(hidden_size, self.num_inputs, num_outputs)
        self.actor_optim = Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(hidden_size, critic_in_size, critic_act_size)
        self.critic_target = Critic(hidden_size, critic_in_size, critic_act_size)
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-3)

        self.gamma = gamma
        self.tau = tau

        hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)


    def select_action(self, state, action_noise=None):
        self.actor.eval()

        mu = self.actor(torch.FloatTensor(state))

        self.actor.train()
        mu = mu.data

        if action_noise is not None:
            mu += torch.Tensor(action_noise.noise())

        return mu


    def train(self, idx, s, a, sn, an, transition, pi_n):
        state_batch = Variable(torch.FloatTensor(s))
        action_batch = Variable(torch.FloatTensor(a))
        reward_batch = Variable(torch.FloatTensor(transition.rewards))
        mask_batch = Variable(torch.FloatTensor(1-np.asarray(transition.dones)))
        next_state_batch = Variable(torch.FloatTensor(sn))

        next_action_batch = Variable(torch.FloatTensor(an))
        next_state_action_values = self.critic_target(next_state_batch, next_action_batch)

        reward_batch = reward_batch.unsqueeze(1)
        
        mask_batch = mask_batch.unsqueeze(1)
        expected_state_action_batch = reward_batch + (self.gamma * mask_batch * next_state_action_values)

        # train critic
        self.critic_optim.zero_grad()

        state_action_batch = self.critic((state_batch), (action_batch))

        value_loss = F.mse_loss(state_action_batch, expected_state_action_batch)
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.1)
        self.critic_optim.step()

        # train actor
        self.actor_optim.zero_grad()

        pol_action_batch = []

        for pi in range(len(pi_n)):
            if pi == idx:
                pol_action_batch.append(self.actor(torch.FloatTensor(transition.states)))
            else:
                pol_action_batch.append(pi_n[idx])

        policy_loss = -self.critic( (state_batch), torch.cat(pol_action_batch, dim=1) )

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.1)
        self.actor_optim.step()

        # update parameters
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()
  