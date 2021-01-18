from torch import nn
from torch.nn import functional as F
from torch import optim
from torch import distributions

import numpy as np
import torch
import itertools


class MLP(nn.Module):
    def __init__(
        self, 
        ob_dim, 
        ac_dim, 
        hidden_dim, 
        n_layers, 
        gamma, 
        discrete,
        learning_rate,
        use_gpu=False
    ):
        super(MLP, self).__init__()
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gamma = gamma
        self.discrete = discrete
        self.learning_rate = learning_rate
        self.device = 'cuda:0' if use_gpu else 'cpu'

        # self.target_qnet = self.build(ob_dim, ac_dim, hidden_dim, n_layers).to(self.device)
        self.qnet = self.build(ob_dim, ac_dim, hidden_dim, n_layers).to(self.device)
        self.optimizer = optim.Adam(self.qnet.parameters(), self.learning_rate)

    def forward(self, obs):
        if self.discrete:
            return self.qnet(obs)

    def get_action(self, obs, epsilon=0.01):
        with torch.no_grad():
            if np.random.random() > epsilon:
                obs = torch.tensor(obs).float().to(self.device)
                action = self.forward(obs).argmax().cpu().numpy()
                return action
            else:
                return np.random.randint(self.ac_dim)

    def build(
        self, 
        in_dim, 
        out_dim, 
        hidden_dim, 
        n_layers,
        activation=nn.ReLU(),
        output_activation=nn.Identity()
    ):
        layers = []
        prev_dim = in_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation)
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, out_dim))
        layers.append(output_activation)
        return nn.Sequential(*layers)

    def update(self, obs, actions, rewards, next_obs):
        obs = torch.tensor(obs).float().to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).float().to(self.device)
        next_obs = torch.tensor(next_obs).float().to(self.device)
        # print(obs, obs.shape)
        # print(actions, actions.shape)
        # print(rewards, rewards.shape)
        # print(next_obs, next_obs.shape)

        q_values = self.qnet(obs)
        with torch.no_grad():
            exp_q_values = q_values.clone()
            target_q_values = self.qnet(next_obs)
            exp_q_values[np.arange(len(q_values)), actions] = rewards + self.gamma * target_q_values.max(dim=1)[0]
        loss = nn.MSELoss()(q_values, exp_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def copy_parameters(self):
        # self.target_qnet.load_state_dict(self.qnet.state_dict())
        pass

