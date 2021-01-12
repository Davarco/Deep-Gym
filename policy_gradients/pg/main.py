from agent import Agent

import numpy as np
import gym
import torch


def cartpole():
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    env.seed(1)
    print('CartPole v0')
    print('Observation Space:', env.observation_space)
    print('Action Space:', env.action_space)
    params = {
        'ob_dim': 4,
        'ac_dim': 2,
        'hidden_dim': 10,
        'n_layers': 3,
        'discrete': True,
        'learning_rate': 0.01,
        'gamma': 0.95,
        'num_rollouts': 16,
        'max_rollout_length': 100000000, # INF, in practice
        'num_episodes': 400,
        'reward_to_go': True,
        'standardize_advantage': True,
        'use_gpu': True
    }
    agent = Agent(env, params) 
    agent.train()
    agent.run_demo()

def inverted_pendulum():
    env = gym.make('InvertedPendulum-v2')
    env = env.unwrapped
    env.seed(1)
    print('Inverted Pendulum v2')
    print('Observation Space:', env.observation_space)
    print('Action Space:', env.action_space)
    params = {
        'ob_dim': 4,
        'ac_dim': 1,
        'hidden_dim': 10,
        'n_layers': 3,
        'learning_rate': 0.01,
        'discrete': False,
        'gamma': 0.95,
        'num_rollouts': 16,
        'max_rollout_length': 100000000, # INF, in practice
        'num_episodes': 400,
        'reward_to_go': True,
        'standardize_advantage': True,
        'use_gpu': True
    }
    agent = Agent(env, params) 
    agent.train()
    agent.run_demo()

def lunar_lander():
    env = gym.make('LunarLanderContinuous-v2')
    env = env.unwrapped
    env.seed(1)
    print('Lunar Lander Continuous v2')
    print('Observation Space:', env.observation_space)
    print('Action Space:', env.action_space)
    params = {
        'ob_dim': 8,
        'ac_dim': 2,
        'hidden_dim': 64,
        'n_layers': 2,
        'learning_rate': 0.005,
        'discrete': False,
        'gamma': 0.99,
        'num_rollouts': 500,
        'max_rollout_length': 40000,
        'num_episodes': 100,
        'reward_to_go': True,
        'standardize_advantage': True,
        'use_gpu': True
    }
    agent = Agent(env, params) 
    agent.train()
    agent.run_demo()

if __name__ == '__main__':
    # cartpole()
    # inverted_pendulum()
    np.random.seed(1)
    torch.manual_seed(1)
    lunar_lander()
