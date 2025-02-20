#from gymnasium import spaces
import random
import numpy as np
import torch

#import Dueling_DQN_Agent.utils.help_classes as hc
import Dueling_DQN_Agent.utils.memory as mem
from Dueling_DQN_Agent.QFunction import QFunction

class Dueling_DQN_Agent(object):

    def __init__(self, state_space, action_space, **userconfig):
        
        """if not isinstance(state_space, spaces.box.Box):
            raise hc.UnsupportedSpace("Observation space {} incompatible with {}. (Require: Box)".format(state_space, self))
        if not isinstance(action_space, spaces.discrete.Discrete):
            raise hc.UnsupportedSpace("Action space {} incompatible with {}. (Reqire Discrete.)".format(action_space, self))"""
        
        self._state_space = state_space
        self._action_space = action_space
        self._action_n = action_space.n

        self.train_iter = 0

        self._config = {
            "eps": 0.05,                       
            "discount": 0.95,
            "buffer_size": int(1e5),
            "batch_size": 256,
            "learning_rate": 0.0002,
            "update_target_every": 20,
            "use_target_net": True,
            "use_eps_decay": False,
            "eps_decay_mode": "exponential",
            "eps_min": 0.01,
            "eps_decay": 0.995,
            "seed": int(random.random()),
            "hidden_sizes": [128,128],
            "value_hidden_sizes": None,
            "advantage_hidden_sizes": None
        }

        self._config.update(userconfig)

        random.seed(self._config["seed"])
        np.random.seed(self._config["seed"])
        torch.manual_seed(self._config["seed"])

        # Ensure deterministic behavior in CUDA if available
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._config["seed"])
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self._eps = self._config["eps"]
        
        self.buffer = mem.Memory(max_size=self._config["buffer_size"])
                
        # Q Network
        self.Q = QFunction(state_dim=self._state_space.shape[0], 
                           action_dim=self._action_n,
                           learning_rate = self._config["learning_rate"],
                           hidden_sizes = self._config["hidden_sizes"],
                           value_hidden_sizes = self._config["value_hidden_sizes"],
                           advantage_hidden_sizes = self._config["advantage_hidden_sizes"])
        
        # Q Target
        self.Q_target = QFunction(state_dim=self._state_space.shape[0], 
                                  action_dim=self._action_n,
                                  learning_rate = 0,    # We do not want to train the Target Function, only copy the weights of the Q Network
                                  hidden_sizes = self._config["hidden_sizes"],
                                  value_hidden_sizes = self._config["value_hidden_sizes"],
                                  advantage_hidden_sizes = self._config["advantage_hidden_sizes"])    # We do not want to train the Target Function, only copy the weights of the Q Network
        self._update_target_net()

    def _update_target_net(self):        
        self.Q_target.load_state_dict(self.Q.state_dict())
    
    def perform_greedy_action(self, state, eps = None):

        if eps is None:
            eps = self._eps

        if np.random.random() > eps:
            action = self.Q.greedyAction(state)
        else: 
            action = self._action_space.sample()
        #print(f"action: {action}, shape: {np.asarray(action).shape}")           
        return action

    def _perform_epsilon_decay(self):

        if self._config["eps_decay_mode"] == "linear":
            self._eps = max(self._config["eps_min"], self._eps - self._config["eps_decay"])
        elif self._config["eps_decay_mode"] == "exponential":
            self._eps = max(self._config["eps_min"], self._eps * self._config["eps_decay"])
        else:
            raise ValueError("Error: Epsilon decay mode must be \"linear\" or \"exponential\".")

    def train(self, iter_fit=32):

        losses = []
        self.train_iter+=1
                     
        for i in range(iter_fit):

            # Sample from the replay buffer
            random.seed(self._config["seed"])
            data = self.buffer.sample(batch=self._config["batch_size"])
            s = np.stack(data[:,0]) # Current state (s_t)
            a = np.stack(data[:,1]) # Action taken (a_t)
            rew = np.stack(data[:,2])[:,None] # Reward received (r)
            s_prime = np.stack(data[:,3]) # Next state (s_t+1)
            done = np.stack(data[:,4])[:,None] # Done flag (1 if terminal, else 0)

            
            if self._config["use_target_net"]:
                v_prime = self.Q_target.maxQ(s_prime)
            else:
                v_prime = self.Q.maxQ(s_prime)

            # target                                              
            td_target = rew + self._config["discount"] * (1.0 - done) * v_prime
            
            # optimize the lsq objective
            fit_loss = self.Q.fit(s, a, td_target)
            
            losses.append(fit_loss)

        if self._config["use_target_net"] and self.train_iter % self._config["update_target_every"] == 0:
            self._update_target_net()
                
        return losses