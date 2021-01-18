from agent import Agent

import numpy as np
import gym
import torch


def cartpole():
    env = gym.make('CartPole-v1')
    env = env.unwrapped
    seed(env, 1)
    print('CartPole v0')
    print('Observation Space:', env.observation_space)
    print('Action Space:', env.action_space)
    params = {
        'ob_dim': 4,
        'ac_dim': 2,
        'hidden_dim': 12,
        'n_layers': 2,
        'discrete': True,
        'cnn': False,
        'buffer_size': 50000,
        'batch_size': 200,
        'max_rollout_length': 1000,
        'num_episodes': 1000,
        'update_freq': 10,
        'learning_rate': 0.01,
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
