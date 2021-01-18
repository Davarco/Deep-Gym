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
        self.max_rollout_length = params['max_rollout_length']
        self.num_episodes = params['num_episodes']
        self.update_freq = params['update_freq']

    def run_demo(self):
        while True:
            self.run_single_iteration()

    def run_single_iteration(self):
        obs = self.env.reset()
        for _ in range(self.max_rollout_length):
            self.env.render()
            action = self.policy.get_action(obs)
            obs, reward, done, _ = self.env.step(action)
            if done: 
                break

    def train(self):
        for i in range(self.num_episodes):
            if i % 50 == 0:
            #     self.policy.copy_parameters()
                self.run_single_iteration()
            eps = max(0.01, 1 - 1/max(i, 300)) 
            rollouts = sample_n_rollouts(self.env, self.policy, self.batch_size, self.max_rollout_length, eps)
            num_rollouts = len(rollouts['reward'])
            total_reward = sum([np.sum(rewards) for rewards in rollouts['reward']])
            rollouts['reward'] = np.concatenate(rollouts['reward'])
            for m in range(rollouts['reward'].size):
                sample = { 
                    'obs': rollouts['obs'][m],
                    'action': rollouts['action'][m],
                    'reward': rollouts['reward'][m],
                    'next_obs': rollouts['next_obs'][m]
                }
                self.buffer.add_sample(sample)
            batch = self.buffer.sample_batch(self.batch_size)
            self.policy.update(batch['obs'], batch['action'], batch['reward'], batch['next_obs'])
            print('Episode {}: Average Reward={}'.format(i, total_reward/num_rollouts))

def sample_rollout(env, policy, max_rollout_length, epsilon=0.01):
    rollout = {
        'obs': [],
        'action': [],
        'reward': [],
        'next_obs': [],
    }
    obs = env.reset()
    for i in range(max_rollout_length):
        rollout['obs'].append(obs)
        action = policy.get_action(obs, epsilon)
        rollout['action'].append(action)
        obs, reward, done, _ = env.step(action)
        rollout['reward'].append(reward)
        rollout['next_obs'].append(obs)
        if done:
            break
        if i == max_rollout_length-1:
            print('Reached max rollout length.')
    for key in rollout:
        rollout[key] = np.array(rollout[key])
    return rollout

def sample_n_rollouts(env, policy, num_steps, max_rollout_length, epsilon=0.01):
    size = 0
    rollouts = dict()
    while size < num_steps:
        rollout = sample_rollout(env, policy, max_rollout_length, epsilon)
        if not rollouts:
            for key in rollout:
                if key != 'reward':
                    rollouts[key] = rollout[key]
                else:
                    rollouts[key] = [ rollout[key] ]
        else:
            for key in rollout:
                if key != 'reward':
                    rollouts[key] = np.concatenate((rollouts[key], rollout[key]))
                else:
                    rollouts[key].append(rollout[key])
        size += rollout['reward'].size
    return rollouts

