import numpy as np

class RandomAgent():

    def __init__(self, seed):
        np.random.seed(seed)

    def act(self, obs):
        return np.random.uniform(-1,1,4)