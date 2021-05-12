from collections import deque, namedtuple
import numpy as np
import random
import torch

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed):
        self.deque = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])

    def add(self, state, action, reward, next_state, done):
        state = np.concatenate(state, axis=0)
        next_state = np.concatenate(next_state, axis=0)
        action = np.concatenate(action, axis=0)

        e = self.experience(state, action, reward, next_state, done)
        self.deque.append(e)

    def sample(self):
        """sample from the buffer"""
        experiences = random.sample(self.deque, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.deque)
