from policy import Policy

import numpy as np
import torch
from collections import defaultdict


class Agent():
    def __init__(self, env, params):
        self.env = env
        self.policy = Policy(
            params['ob_dim'],
            params['ac_dim'],
            params['hidden_dim'],
            params['n_layers'],
            params['discrete'],
            params['learning_rate'],
            params['use_gpu']
        )
        self.gamma = params['gamma']
        self.batch_size = params['batch_size']
        self.max_rollout_length = params['max_rollout_length']
        self.num_episodes = params['num_episodes']
        self.reward_to_go = params['reward_to_go']
        self.standardize_advantage = params['standardize_advantage']

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
            avg_reward = self.train_one_iteration()
            print('Itr {}: Average Reward={}'.format(i, avg_reward))
            if i % 10 == 0:
                self.run_single_iteration()
            # if avg_reward > 1000:
            #    break

    def train_one_iteration(self):
        rollouts = sample_n_rollouts(self.env, self.policy, self.batch_size, self.max_rollout_length)
        num_rollouts = len(rollouts['reward'])
        total_reward = sum([np.sum(rewards) for rewards in rollouts['reward']])
        advantages = self.estimate_advantages(rollouts['reward'])
        self.policy.update(rollouts['obs'], rollouts['action'], advantages)
        return total_reward/num_rollouts
    
    def estimate_advantages(self, rewards_list):
        ret = None
        for rewards in rewards_list:
            T = len(rewards)
            if not self.reward_to_go:
                p = np.arange(T)
                q = np.sum(np.multiply(np.power(self.gamma, p), rewards)) 
                temp = np.array([q for _ in range(T)])
                ret = temp if ret is None else np.concatenate((ret, temp))
            else:
                temp = np.zeros(T)
                for i in range(T):
                    p = np.arange(T-i)
                    q = np.sum(np.multiply(np.power(self.gamma, p), rewards[i:])) 
                    temp[i] = q 
                ret = temp if ret is None else np.concatenate((ret, temp))
        if self.standardize_advantage:
            ret -= np.mean(ret)
            ret /= np.std(ret)
        return ret

def sample_rollout(env, policy, max_rollout_length):
    rollout = {
        'obs': [],
        'action': [],
        'reward': [],
        'next_obs': [],
        'render': []
    }
    obs = env.reset()
    for i in range(max_rollout_length):
        rollout['obs'].append(obs)
        action = policy.get_action(obs)
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

def sample_n_rollouts(env, policy, batch_size, max_rollout_length):
    size = 0
    rollouts = dict()
    while size < batch_size:
        rollout = sample_rollout(env, policy, max_rollout_length)
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

