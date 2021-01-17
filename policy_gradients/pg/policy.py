from torch import nn
from torch.nn import functional as F
from torch import optim
from torch import distributions

import numpy as np
import torch
import itertools


class Policy(nn.Module):
    def __init__(
        self, 
        ob_dim, 
        ac_dim, 
        hidden_dim, 
        n_layers, 
        discrete,
        learning_rate,
        use_gpu=False
    ):
        super(Policy, self).__init__()
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.learning_rate = learning_rate
        self.device = 'cuda:0' if use_gpu else 'cpu'

        if self.discrete:
            self.nn = self.build(ob_dim, ac_dim, hidden_dim, n_layers).to(self.device)
            self.optimizer = optim.Adam(self.nn.parameters(), self.learning_rate)
        else:
            self.mean = self.build(ob_dim, ac_dim, hidden_dim, n_layers).to(self.device)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=self.device)
            )
            self.optimizer = optim.Adam(
                list(self.mean.parameters()) + [self.logstd],
                self.learning_rate
            )

    def forward(self, obs):
        if self.discrete:
            logits = self.nn(obs)
            return distributions.Categorical(logits=logits)
        else:
            mean = self.mean(obs)
            return distributions.MultivariateNormal(mean, scale_tril=torch.diag(self.logstd.exp()))

    def get_action(self, obs):
        obs = torch.tensor(obs).float().to(self.device)
        dis = self.forward(obs)
        return dis.sample().cpu().numpy()

    def build(
        self, 
        in_dim, 
        out_dim, 
        hidden_dim, 
        n_layers,
        activation=nn.Tanh(),
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
        # print(obs, type(obs), obs.shape)
        # print(actions, type(actions), actions.shape)
        # print(advantages, type(advantages), advantages.shape)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

