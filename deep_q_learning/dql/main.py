from agent import Agent

import numpy as np
import gym
import torch


def cartpole():
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    seed(env, 1)
    print('CartPole v0')
    print('Observation Space:', env.observation_space)
    print('Action Space:', env.action_space)
    params = {
        'ob_dim': 4,
        'ac_dim': 2,
        'hidden_dim': 32,
        'n_layers': 3,
        'discrete': True,
        'cnn': False,
        'buffer_size': 50000,
        'batch_size': 128,
        'num_episodes': 1000,
        'episode_length': 1000,
        'update_freq': 100,
        'learning_rate': 0.0001,
        'gamma': 0.99,
        'use_gpu': True
    }
    agent = Agent(env, params) 
    agent.train()
    agent.run_demo()

def seed(env, i):
    env.seed(i) 
    np.random.seed(1)
    torch.manual_seed(1)

if __name__ == '__main__':
    cartpole()
    # lunar_lander()
