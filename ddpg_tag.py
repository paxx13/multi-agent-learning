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
from ddpg import Agent
from replay_memory import ReplayMemory, Transition
from make_env import make_env
import simple_tag_utilities

MAX_EPISODES = 100000
GAMMA = 0.9
TAU = 0.01
NOISE_SCALE = 0.3

BATCH_SIZE=32


def train_agents(agents, memories, states, actions, rewards, states_next, done, batch_size=BATCH_SIZE):
    num_agents = len(agents)
    v_loss = np.zeros(num_agents)
    p_loss = np.zeros(num_agents)
    for i in range(num_agents):
        if done[i]:
            rewards[i] -= 500

        memories[i].push(states[i], actions[i], rewards[i], states_next[i], not done[i])

        if len(memories[i]) > batch_size * 10:
           transitions = memories[i].sample(batch_size)
           batch = Transition(*zip(*transitions))
           v_loss[i], p_loss[i] = agents[i].train(batch)
            
    return v_loss, p_loss


def train(env, agents, ounoise, memories, render=False, training=False):
    num_agents = len(agents)
    
    writer = SummaryWriter(comment="-multiagent")  
        
    for episode in trange(MAX_EPISODES):
        states = env.reset()
        episode_rewards = np.zeros(num_agents)        
        episode_vlosses = np.zeros(num_agents)       
        episode_plosses = np.zeros(num_agents)
        collision_count = np.zeros(num_agents)
        steps = 0

        for i in range(num_agents):
            ounoise[i].scale = (NOISE_SCALE - 0.1) * max(0, 20000 - episode) / MAX_EPISODES + 0.1
            ounoise[i].reset() 

        while True:
            steps += 1

            if render:
                env.render()

            # act
            actions = []
            for i in range(num_agents):
                
                #action = np.clip(agents[i].select_action(states[i], ounoise[i]), -2, 2)
                action = agents[i].select_action(states[i], ounoise[i])
                actions.append(action.squeeze(0).numpy())
            
            # step
            states_next, rewards, done, _ = env.step(actions)

            # learn
            if training:
                v_loss, p_loss = train_agents(agents, memories, states, actions, rewards, states_next, done)

            states = states_next
            episode_rewards += rewards
            episode_vlosses += v_loss
            episode_plosses += p_loss
            collision_count += np.array(
                simple_tag_utilities.count_agent_collisions(env))

            # reset states if done
            if any(done):
                episode_rewards = episode_rewards / steps
                episode_vlosses = episode_vlosses / steps
                episode_plosses = episode_plosses / steps
                
                for a in range(num_agents):
                    writer.add_scalar('rewards/agent'+str(a),    episode_rewards[a],   episode)
                    writer.add_scalar('vlosses/agent'+str(a),    episode_vlosses[a],   episode)
                    writer.add_scalar('plosses/agent'+str(a),    episode_plosses[a],   episode)
                    writer.add_scalar('collisions/agent'+str(a), collision_count[a],   episode)
                break 
                 
                
    writer.close()   


def play(env, agents):
    num_agents = len(agents)

    for episode in trange(MAX_EPISODES):
        states = env.reset()

        while True:
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
            
            time.sleep(1.0/60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    parser.add_argument('-t', '--train', help='set to learn a policy', action="store_true")
    parser.add_argument('--env', default='simple_tag_guided', type=str)
    parser.add_argument('--memory_size', default=10000, type=int)
    args = parser.parse_args()

    # init env
    env = make_env(args.env)

    # set random seed
    env.seed(2)
    random.seed(2)
    np.random.seed(2)

    agents = []
    noise = []
    memories = []
    for i in range(env.n):
        n_action = env.action_space[i].n
        state_size = env.observation_space[i].shape[0]
        speed = 0.8 if env.agents[i].adversary else 1

        agents.append(Agent(speed, GAMMA, TAU, 50, state_size, n_action) )
        noise.append(OUNoise(n_action))
        memories.append(ReplayMemory(args.memory_size))

    if args.train: 
        try:
            train(env, agents, noise, memories, render=False, training=True)
        except (KeyboardInterrupt, SystemExit):
            print('trainig aborted')
        
        for agent in agents:
            agent.save_model(args.env, actor_path="models/ddpg_actor", critic_path="models/ddpg_critic")
        pickle.dump(agents, open('./models/agents.obj', "wb" ) )
    else:
        #for agent in agents:
            #agent.load_model("models/ddpg_actor", "models/ddpg_critic")
        agents =  pickle.load(open('./models/agents.obj', "rb")) 
        play(env, agents)
