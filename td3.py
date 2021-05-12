# individual network settings for each actor + critic pair
# see model for details

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
import numpy as np

# add OU noise for exploration
from OUNoise import OUNoise
from model import Actor, Critic

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

td3_config = {
    'td3_noise': 0.2,
    'td3_noise_clip': 0.5,
    'td3_delay': 2
}


class TD3Agent:
    # Shared critic
    critic1 = None
    target_critic1 = None
    critic_optimizer1 = None

    critic2 = None
    target_critic2 = None
    critic_optimizer2 = None

    def __init__(self, agent_number, state_size, action_size, config):
        self.agent_number = agent_number
        self.state_size = state_size
        self.action_size = action_size

        self.batch_size = config['BATCH_SIZE']
        self.gamma = config['GAMMA']
        self.tau = config['TAU']

        # initialize Actor Network
        self.actor = Actor(state_size, action_size).to(device)
        self.target_actor = Actor(state_size, action_size).to(device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=config['LR_ACTOR'])

        # initialize Critic Networks
        if TD3Agent.critic1 is None:
            TD3Agent.critic1 = Critic(state_size, action_size).to(device)
        if TD3Agent.target_critic1 is None:
            TD3Agent.target_critic1 = Critic(state_size, action_size).to(device)
            TD3Agent.target_critic1.load_state_dict(TD3Agent.critic1.state_dict())
        if TD3Agent.critic_optimizer1 is None:
            TD3Agent.critic_optimizer1 = Adam(TD3Agent.critic1.parameters(), lr=config['LR_CRITIC'])
        self.critic1 = TD3Agent.critic1
        self.target_critic1 = TD3Agent.target_critic1
        self.critic_optimizer1 = TD3Agent.critic_optimizer1

        if TD3Agent.critic2 is None:
            TD3Agent.critic2 = Critic(state_size, action_size).to(device)
        if TD3Agent.target_critic2 is None:
            TD3Agent.target_critic2 = Critic(state_size, action_size).to(device)
            TD3Agent.target_critic2.load_state_dict(TD3Agent.critic2.state_dict())
        if TD3Agent.critic_optimizer2 is None:
            TD3Agent.critic_optimizer2 = Adam(TD3Agent.critic2.parameters(), lr=config['LR_CRITIC'])
        self.critic2 = TD3Agent.critic2
        self.target_critic2 = TD3Agent.target_critic2
        self.critic_optimizer2 = TD3Agent.critic_optimizer2

        self.noise = OUNoise(action_size)

        self.total_steps = 0

    def act(self, obs, add_noise=True):
        obs = torch.from_numpy(obs).float().to(device)
        self.actor.eval() # Set the local policy in evaluation mode
        with torch.no_grad():
            action = self.actor(obs).cpu().data.numpy()
        self.actor.train() # Set the local policy in training mode
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def step(self, memory):
        self.total_steps += 1
        if len(memory) > self.batch_size:
            experiences = memory.sample()
            self.learn(experiences, self.gamma)

    def learn(self, experiences, gamma):
        x, actions, rewards, next_x, dones = experiences

        states = torch.chunk(x, 2, dim=1)
        next_states = torch.chunk(next_x, 2, dim=1)
        rewards = rewards[:, self.agent_number].reshape(rewards.shape[0], 1)
        dones = dones[:, self.agent_number].reshape(dones.shape[0], 1)

        # --- update critic ---
        actions_next = [self.target_actor(x) for x in next_states]
        actions_next_ = torch.cat(actions_next, dim=1).to(device)
        # Generate random noise
        noise = torch.randn_like(actions_next_).mul(td3_config['td3_noise'])
        noise = noise.clamp(-td3_config['td3_noise_clip'], td3_config['td3_noise_clip'])

        actions_next_ = (actions_next_ + noise).clamp(-1, 1)

        Q1_targets_next = self.target_critic1(next_x, actions_next_)
        Q2_targets_next = self.target_critic2(next_x, actions_next_)

        Q_targets_next = torch.min(Q1_targets_next, Q2_targets_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_targets = Q_targets.detach()
        # Compute critic loss
        Q1_expected = self.critic1(x, actions)
        Q2_expected = self.critic2(x, actions)

        critic_loss1 = F.mse_loss(Q1_expected, Q_targets)
        critic_loss2 = F.mse_loss(Q2_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer1.zero_grad()
        critic_loss1.backward()
        self.critic_optimizer1.step()

        self.critic_optimizer2.zero_grad()
        critic_loss2.backward()
        self.critic_optimizer2.step()

        if (self.total_steps) % td3_config['td3_delay']:
            # --- update actor ---
            actions_pred = [self.actor(s) for s in states]
            actions_pred_ = torch.cat(actions_pred, dim=1).to(device)
            # actor_loss = -self.critic1(x, actions_pred_)[self.agent_number].mean()
            actor_loss = -self.critic1(x, actions_pred_).mean()
            # Minimize the loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # update target newtorks
            self.soft_update(self.critic1, self.target_critic1, self.tau)
            self.soft_update(self.critic2, self.target_critic2, self.tau)
            self.soft_update(self.actor, self.target_actor, self.tau)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
