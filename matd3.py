import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f

from buffer import ReplayBuffer
from td3 import TD3Agent

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'


class MATD3:
    def __init__(self, state_size, action_size, num_agents, config):
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents

        self.agents = [TD3Agent(i, state_size, action_size, config) for i in range(num_agents)]
        self.memory = ReplayBuffer(config['BUFFER_SIZE'], config['BATCH_SIZE'], seed=config['SEED'])

    def step(self, states, actions, rewards, next_states, dones):
        self.memory.add(states, actions, rewards, next_states, dones)

        for agent in self.agents:
            agent.step(self.memory)

    def act(self, states, add_noise=True):
        actions = np.zeros((self.num_agents, self.action_size))
        for i, agent in enumerate(self.agents):
            actions[i, :] = agent.act(states[i], add_noise)
        return actions

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def save_model(self):
        for index, agent in enumerate(self.agents):
            torch.save(agent.actor.state_dict(), f'agent{index + 1}_checkpoint_actor.pth')
            torch.save(agent.critic1.state_dict(), f'agent{index + 1}_checkpoint_critic1.pth')
            torch.save(agent.critic2.state_dict(), f'agent{index + 1}_checkpoint_critic2.pth')
