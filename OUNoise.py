import copy
import numpy as np

OU_THETA = 0.15         # how "strongly" the system reacts to perturbations
OU_SIGMA = 0.2          # the variation or the size of the noise

# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class OUNoise:

    def __init__(self, action_dimension, seed=0, mu=0, theta=OU_THETA, sigma=OU_SIGMA):
        self.mu = mu * np.ones(action_dimension)
        self.theta = theta
        self.sigma = sigma
        self.seed = np.random.seed(seed)
        self.action_dimension = action_dimension
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        w = self.sigma * (np.random.standard_normal(self.action_dimension))
        dx = self.theta * (self.mu - x) + w
        self.state = x + dx
        return self.state
