import gym
import numpy as np
import argparse
import os
import pickle
import random
from tqdm import trange
import time

from tensorboardX import SummaryWriter

from ounoise import OUNoise
from maddpg import Agent
from replay_memory import ReplayMemory, Transition
from multiagent_envs.make_env import make_env


from collections import namedtuple

MAX_STEPS = 100
MAX_EPISODES = 10000
GAMMA = 0.96
TAU = 0.01

EXPLORE_NOISE_START = 0.1
EXPLORE_NOISE_FINAL = 0.001
EXPLORE_EPISODES = MAX_EPISODES  * 0.2

BATCH_SIZE=64


def train_agents(agents, memories, batch_size=BATCH_SIZE):
    num_agents = len(agents)
    v_loss = np.zeros(num_agents)
    p_loss = np.zeros(num_agents)

    size = memories[0].position

    # start training once enough data is available 
    if size > batch_size * 10:        
        batch = random.sample(range(size), batch_size)
        s_n = []
        a_n = []
        sn_n = []
        an_n = []
        pi_acts_n = []

        # create training data for critics with states and actions of each agent
        for i in range(num_agents):
            transition = memories[i].sample(batch)
            transition = Transition(*zip(*transition))
            
            s_n.append(transition.states)
            a_n.append(transition.actions)            
            sn_n.append(transition.next_states)
            an_n.append(agents[i].select_action(transition.next_states))
            pi_acts_n.append(agents[i].select_action(transition.states))
            
        s_n = np.concatenate(s_n, axis=1)
        a_n = np.concatenate(a_n, axis=1)
        sn_n = np.concatenate(sn_n, axis=1)        
        an_n = np.concatenate(an_n, axis=1)
        
        # train each agent
        for i in range(num_agents):
            transition = memories[i].sample(batch)
            transition = Transition(*zip(*transition))
            v_loss[i], p_loss[i] = agents[i].train(i, s_n, a_n, sn_n, an_n, transition, pi_acts_n)
            
    return v_loss, p_loss


def train(env, agents, ounoise, memories):
    num_agents = len(agents)

    writer = SummaryWriter(comment="-multiagent")  

    for episode in trange(MAX_EPISODES):
        states = env.reset()
        episode_rewards = np.zeros(num_agents)
        episode_vlosses = np.zeros(num_agents)
        episode_plosses = np.zeros(num_agents)

        # decay exploration noise
        for i in range(num_agents):
            ounoise[i].scale = (EXPLORE_NOISE_START - EXPLORE_NOISE_FINAL) * max(0, EXPLORE_EPISODES - episode) / EXPLORE_EPISODES + EXPLORE_NOISE_FINAL
            ounoise[i].reset() 

        for steps in range(MAX_STEPS):

            # act
            actions = []
            for i in range(num_agents):
                action = agents[i].select_action(states[i], action_noise=ounoise[i])
                actions.append(action.squeeze(0).numpy())            

            # step
            states_next, rewards, done, _ = env.step(actions)

            # save experiences
            for i in range(num_agents):
                memories[i].push(states[i], actions[i], rewards[i], states_next[i], done[i])

            # learn
            v_loss, p_loss = train_agents(agents, memories)

            states = states_next
            
            episode_rewards += rewards
            episode_vlosses += v_loss
            episode_plosses += p_loss

            # reset states if done
            if any(done) or steps==MAX_STEPS-1:
                if steps > 0:
                    episode_rewards = episode_rewards / steps
                    episode_vlosses = episode_vlosses / steps
                    episode_plosses = episode_plosses / steps

                # logging
                for a in range(num_agents):
                    writer.add_scalar('rewards/agent'+str(a),    episode_rewards[a],   episode)
                    writer.add_scalar('vlosses/agent'+str(a),    episode_vlosses[a],   episode)
                    writer.add_scalar('plosses/agent'+str(a),    episode_plosses[a],   episode)                 
                    writer.add_scalar('exploration/agent'+str(a)+' scale', ounoise[a].scale, episode)

                break 

    writer.close()   


def play(env, agents):
    num_agents = len(agents)

    for episode in trange(MAX_EPISODES):
        states = env.reset()

        for steps in range(MAX_STEPS):
            env.render()

            # act
            actions = []
            for i in range(env.n):
                action = agents[i].select_action(states[i])
                actions.append(action.squeeze(0).numpy())

            # step
            states, rewards, done, _ = env.step(actions)
            if any(done):
                break

            time.sleep(1.0/10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    parser.add_argument('-t', '--train', help='set to learn a policy', action="store_true")
    parser.add_argument('--env', help='name of the environment', default='simple_tag', type=str)
    parser.add_argument('--memory_size', help='size of the replay buffer', default=1000000, type=int)
    args = parser.parse_args()

    # init env
    env = make_env(args.env)

    # set random seed
    env.seed(2)
    random.seed(2)
    np.random.seed(2)

    # define the size of the input dimensions for the centralized critics (information from all agents)
    critic_in_size = 0
    critic_act_size = 0

    for i in range(env.n):
        critic_in_size += env.observation_space[i].shape[0]
        critic_act_size += env.action_space[i].n


    # create agents, replay buffers and exploration noise
    agents = []
    noise = []
    memories = []    

    for i in range(env.n):
        n_action = env.action_space[i].n
        state_size = env.observation_space[i].shape[0]

        agents.append(Agent(GAMMA, TAU, 50, state_size, n_action, critic_in_size, critic_act_size) )
        noise.append(OUNoise(n_action))
        memories.append(ReplayMemory(args.memory_size))

    if args.train: 
        try:
            train(env, agents, noise, memories)
        except (KeyboardInterrupt, SystemExit):
            print('trainig aborted')
        
        pickle.dump(agents, open('./models/agents_'+args.env+'.obj', "wb" ) )
    else:
        agents =  pickle.load(open('./models/agents_'+args.env+'.obj', "rb")) 
        play(env, agents)
