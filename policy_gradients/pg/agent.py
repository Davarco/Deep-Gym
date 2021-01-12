from policy import Policy
import numpy as np
import torch


class Agent():
    def __init__(self, env, params):
        self.env = env
        self.actor = Policy(
            params['ob_dim'],
            params['ac_dim'],
            params['hidden_dim'],
            params['n_layers'],
            params['discrete'],
            params['learning_rate'],
            params['use_gpu']
        )
        self.gamma = params['gamma']
        self.num_rollouts = params['num_rollouts']
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
            action = self.actor.get_action(obs)
            obs, reward, done, _ = self.env.step(action)
            if done: 
                break

    def train(self):
        best_avg_reward = -float('inf')
        for i in range(self.num_episodes):
            avg_reward = self.train_one_iteration()/self.num_rollouts
            best_avg_reward = max(best_avg_reward, avg_reward)
            print('Itr {}: Average Reward={}, Best Reward={}'.format(i, avg_reward, best_avg_reward))
            if i % 100 == 0:
                self.run_single_iteration()
            # if avg_reward > 1000:
            #    break

    def train_one_iteration(self):
        rollouts = sample_n_rollouts(self.env, self.actor, self.num_rollouts, self.max_rollout_length)
        q_values = self.estimate_q_values(rollouts['reward'])
        advantages = self.estimate_advantages(q_values)
        total_reward = sum([np.sum(rewards) for rewards in rollouts['reward']])
        self.actor.update(rollouts['obs'], rollouts['action'], advantages)
        return total_reward
    
    def estimate_q_values(self, rewards_list) -> np.ndarray:
        ret = None
        for rewards in rewards_list:
            if not self.reward_to_go:
                q, c = 0, 1
                for r in rewards: 
                    q += c*r
                    c *= self.gamma
                temp = np.array([q for _ in range(len(rewards_list))])
                ret = temp if ret is None else np.concatenate((ret, temp))
            else:
                temp = np.zeros(len(rewards))
                q, c = 0, 1
                for i in reversed(range(len(rewards))):
                    q += c * rewards[i]
                    c *= self.gamma
                    temp[i] = q
                ret = temp if ret is None else np.concatenate((ret, temp))
        return ret

    def estimate_advantages(self, q_values):
        if self.standardize_advantage:
            q_values -= np.mean(q_values)
            q_values /= np.std(q_values)
        return q_values

def sample_rollout(env, actor, max_rollout_length) -> np.ndarray:
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
        action = actor.get_action(obs)
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

def sample_n_rollouts(env, actor, num_rollouts, max_rollout_length) -> np.ndarray:
    rollouts = dict()
    for i in range(num_rollouts):
        rollout = sample_rollout(env, actor, max_rollout_length)
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
    return rollouts


