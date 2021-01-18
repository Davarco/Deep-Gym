import numpy as np


class ReplayBuffer():
    def __init__(self, buffer_size, ob_dim, ac_dim):
        self.buffer = {
            'obs': np.zeros((buffer_size, ob_dim)),
            'action': np.zeros(buffer_size, dtype=int),
            'reward': np.zeros(buffer_size),
            'next_obs': np.zeros((buffer_size, ob_dim)),
            'done': np.zeros(buffer_size, dtype=int)
        }
        self.buffer_size = buffer_size
        self.index = 0
        self.count = 0

    def sample_batch(self, batch_size):
        assert self.count >= batch_size
        batch = dict()
        rows = np.random.choice(self.count, batch_size, replace=False)
        for key in self.buffer:
            batch[key] = self.buffer[key][rows]
        return batch

    def add_sample(self, sample):
        for key in sample:
            self.buffer[key][self.index] = sample[key]
        self.index = (self.index + 1) % self.buffer_size
        self.count = min(self.buffer_size, self.count+1)
    
