from policy import MLP
from replay_buffer import ReplayBuffer

import numpy as np
import torch
from collections import defaultdict


class Agent():
    def __init__(self, env, params):
        self.env = env
        self.buffer = ReplayBuffer(params['buffer_size'], params['ob_dim'], params['ac_dim'])
        if not params['cnn']:
            self.policy = MLP(
                params['ob_dim'],
                params['ac_dim'],
                params['hidden_dim'],
                params['n_layers'],
                params['gamma'],
                params['discrete'],
                params['learning_rate'],
                params['use_gpu']
            )
        self.batch_size = params['batch_size']
        self.num_episodes = params['num_episodes']
        self.episode_length = params['episode_length']
        self.update_freq = params['update_freq']

    def run_demo(self):
        while True:
            self.run_single_iteration()

    def run_single_iteration(self, render=True):
        obs = self.env.reset()
        total_reward = 0
        for _ in range(self.episode_length):
            if render:
                self.env.render()
            action = self.policy.get_action(obs)
            obs, reward, done, _ = self.env.step(action)
            total_reward += reward
            if done: 
                return total_reward

    def train(self):
        obs = self.env.reset()
        for i in range(self.num_episodes):
            if i % self.update_freq == 0:
                self.policy.copy_parameters()
            if i % 20 == 0:
                reward = self.run_single_iteration()
                obs = None
            # Some form of epsilon annealing, though this might not be the best.
            eps = max(0.01, 1 - min(i, 500)/500) 
            total_loss = 0
            
            for t in range(self.episode_length): 
                obs = self.env.reset() if obs is None else obs
                action = self.policy.get_action(obs, eps)
                # print(obs, action)
                next_obs, reward, done, _ = self.env.step(action)
                sample = { 'obs': obs, 'action': action, 'reward': reward, 'next_obs': next_obs, 'done': done }
                self.buffer.add_sample(sample)
                obs = next_obs

                if self.buffer.count >= self.batch_size:
                    b = self.buffer.sample_batch(self.batch_size)
                    total_loss += self.policy.update(b['obs'], b['action'], b['reward'], b['next_obs'], b['done'])
                if done:
                    obs = None
                    print('[Episode {}]: Reward={} Loss={}'.format(i, t+1, total_loss/(t+1)))
                    break


