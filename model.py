import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

actor_net = {'fc1_units': 256, 'fc2_units': 128}
critic_net = {'fc1_units': 256, 'fc2_units': 128}


class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed=0,
                 fc1_units=actor_net['fc1_units'],
                 fc2_units=actor_net['fc2_units']):
        super(Actor, self).__init__()

        """self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_norm.weight.data.fill_(1)
        self.input_norm.bias.data.fill_(0)"""

        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.nonlin = F.relu #leaky_relu
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x):
        h1 = self.nonlin(self.fc1(x))
        h2 = self.nonlin(self.fc2(h1))
        h3 = F.tanh(self.fc3(h2))
        return h3


class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed=0,
                 fc1_units=critic_net['fc1_units'],
                 fc2_units=critic_net['fc2_units']):
        super(Critic, self).__init__()

        """self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_norm.weight.data.fill_(1)
        self.input_norm.bias.data.fill_(0)"""

        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size * 2, fc1_units)
        self.fc2 = nn.Linear(fc1_units + action_size * 2, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.nonlin = F.leaky_relu
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        # critic network simply outputs a number
        xs = self.nonlin(self.fc1(state))
        h1 = torch.cat((xs, action), dim=1)
        h2 = self.nonlin(self.fc2(h1))
        h3 = self.fc3(h2)
        return h3
