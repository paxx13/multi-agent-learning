import gym
import numpy as np
import argparse
import os
import pickle
import code
import random
from tqdm import trange

from tensorboardX import SummaryWriter

from ounoise import OUNoise
from ddpg import Agent
from replay_memory import ReplayMemory, Transition
from make_env import make_env
import simple_tag_utilities

MAX_EPISODES = 1000000
GAMMA = 0.9
TAU = 0.01
NOISE_SCALE = 0.3

BATCH_SIZE=32


def train_agents(agents, states, actions, rewards, states_next, done, batch_size=BATCH_SIZE):
    v_loss = np.zeros(len(agents) )
    p_loss = np.zeros(len(agents) )
    for i in range(len(agents)):
        if done[i]:
            rewards[i] -= 500

        memories[i].push(states[i], actions[i], rewards[i], states_next[i], not done[i])

        if len(memories[i]) > batch_size * 10:
           transitions = memories[i].sample(batch_size)
           batch = Transition(*zip(*transitions))
           v_loss[i], p_loss[i] = agents[i].train(batch)
            
    return v_loss, p_loss

def train(agents, ounoise, checkpoint_interval, weights_filename_prefix, csv_filename_prefix, render=False, training=False):
    num_agents = len(agents)
    
    writer = SummaryWriter(comment="-multiagent")  
        
    for episode in trange(MAX_EPISODES):
        states = env.reset()
        episode_rewards = np.zeros(num_agents)        
        episode_vlosses = np.zeros(num_agents)       
        episode_plosses = np.zeros(num_agents)
        collision_count = np.zeros(num_agents)
        steps = 0

        for i in range(env.n):
            ounoise[i].scale = (NOISE_SCALE - 0.1) * max(0, 20000 - episode) / 100 + 0.1
            ounoise[i].reset() 

        while True:
            steps += 1

            if render:
                env.render()

            # act
            actions = []
            for i in range(env.n):
                
                #action = np.clip(agents[i].select_action(states[i], ounoise[i]), -2, 2)
                action = agents[i].select_action(states[i], ounoise[i])
                actions.append(action.squeeze(0).numpy())
            
            # step
            states_next, rewards, done, _ = env.step(actions)

            # learn
            if training:
                v_loss, p_loss = train_agents(agents, states, actions, rewards, states_next, done)

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
                
                for a in range(env.n):
                    writer.add_scalar('rewards/agent'+str(a),    episode_rewards[a],   episode)
                    writer.add_scalar('vlosses/agent'+str(a),    episode_vlosses[a],   episode)
                    writer.add_scalar('plosses/agent'+str(a),    episode_plosses[a],   episode)
                    writer.add_scalar('collisions/agent'+str(a), collision_count[a],   episode)
                break 
                 
                
    writer.close()   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='simple_tag_guided', type=str)
    parser.add_argument('--video_dir', default='videos/', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--video_interval', default=1000, type=int)
    parser.add_argument('--render', default=False, action="store_true")
    parser.add_argument('--experiment_prefix', default=".",
                        help="directory to store all experiment data")
    parser.add_argument('--weights_filename_prefix', default='/save/tag-ddpg',
                        help="where to store/load network weights")
    parser.add_argument('--csv_filename_prefix', default='/save/statistics-ddpg',
                        help="where to store statistics")
    parser.add_argument('--checkpoint_frequency', default=500, type=int,
                        help="how often to checkpoint")
    parser.add_argument('--testing', default=False, action="store_true",
                        help="reduces exploration substantially")
    parser.add_argument('--load_weights_from_file', default='',
                        help="where to load network weights")
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


    if args.load_weights_from_file != "":
        saver.restore(session, args.load_weights_from_file)
        print("restoring from checkpoint {}".format(
            args.load_weights_from_file))


    # play
    try:
        train(agents, noise, args.checkpoint_frequency,
                       args.experiment_prefix + args.weights_filename_prefix,
                       args.experiment_prefix + args.csv_filename_prefix,
                       render=args.render, training=True)
    except (KeyboardInterrupt, SystemExit):
        print('trainig aborted')
    
    for agent in agents:
        agent.save_model
