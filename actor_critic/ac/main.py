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
        'hidden_dim': 10,
        'n_layers': 2,
        'discrete': True,
        'learning_rate': 0.01,
        'gamma': 0.95,
        'batch_size': 1000,
        'max_rollout_length': 100000000, # INF, in practice
        'num_episodes': 100,
        'standardize_advantage': True,
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
    # inverted_pendulum()
    # lunar_lander()
