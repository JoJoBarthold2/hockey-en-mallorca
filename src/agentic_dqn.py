import gymnasium as gym
from gymnasium import spaces
import numpy as np
import itertools
import time
import torch
import pylab as plt

# %matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from segment_tree import SumSegmentTree, MinSegmentTree
import numpy.random as random
import logging
import time

# class to store transitions
class Memory:
    def __init__(self, max_size=100000):
        self.transitions = np.asarray([])
        self.size = 0
        self.current_idx = 0
        self.max_size = max_size

    def add_transition(self, transitions_new):
        if self.size == 0:
            blank_buffer = [np.asarray(transitions_new, dtype=object)] * self.max_size
            self.transitions = np.asarray(blank_buffer)

        self.transitions[self.current_idx, :] = np.asarray(
            transitions_new, dtype=object
        )
        self.size = min(self.size + 1, self.max_size)
        self.current_idx = (self.current_idx + 1) % self.max_size

    def sample(self, batch=1):
        if batch > self.size:
            batch = self.size
        self.inds = np.random.choice(range(self.size), size=batch, replace=False)
        return self.transitions[self.inds, :]

    def get_all_transitions(self):
        return self.transitions[0 : self.size]


class DQNAgent(object):
    """
    Agent implementing Q-learning with NN function approximation.
    """

    def __init__(self, observation_space, action_space, mem, **userconfig):

        if not isinstance(observation_space, spaces.box.Box):
            raise UnsupportedSpace(
                "Observation space {} incompatible "
                "with {}. (Require: Box)".format(observation_space, self)
            )
        if not isinstance(action_space, spaces.discrete.Discrete):
            raise UnsupportedSpace(
                "Action space {} incompatible with {}."
                " (Reqire Discrete.)".format(action_space, self)
            )
        self.mem = mem
        self._observation_space = observation_space
        self._action_space = action_space
        self._action_n = action_space.n
        self._config = {
            "eps": 0.05,  # Epsilon in epsilon greedy policies
            "discount": 0.95,
            "buffer_size": int(1e5),
            "batch_size": 128,
            "learning_rate": 0.0002,
            "update_target_every": 20,
            "use_target_net": True,
        }
        self._config.update(userconfig)
        self._eps = self._config["eps"]
        self.sample_times = []

        self.buffer = self.mem(max_size=self._config["buffer_size"])

        # Q Network
        self.Q = QFunction(
            observation_dim=self._observation_space.shape[0],
            action_dim=self._action_n,
            learning_rate=self._config["learning_rate"],
        )
        # Q Network
        self.Q_target = QFunction(
            observation_dim=self._observation_space.shape[0],
            action_dim=self._action_n,
            learning_rate=0,
        )
        self._update_target_net()
        self.train_iter = 0

    def _update_target_net(self):
        self.Q_target.load_state_dict(self.Q.state_dict())

    def act(self, observation, eps=None):
        if eps is None:
            eps = self._eps
        # epsilon greedy
        if np.random.random() > eps:
            action = self.Q.greedyAction(observation)
        else:
            action = self._action_space.sample()
        return action

    def store_transition(self, transition):
        self.buffer.add_transition(transition)

    def train(self, iter_fit=32):
        losses = []
        self.train_iter += 1
        if (
            self._config["use_target_net"]
            and self.train_iter % self._config["update_target_every"] == 0
        ):
            self._update_target_net()
        for i in range(iter_fit):

            # sample from the replay buffer
            start_time = time.time()
            data = self.buffer.sample(batch=self._config["batch_size"])
            #print(f" Sample time: {time.time()- start_time}")
            self.sample_times.append(time.time()- start_time)
            s = np.stack(data[:, 0])  # s_t
            a = np.stack(data[:, 1])  # a_t
            rew = np.stack(data[:, 2])[:, None]  # rew  (batchsize,1)
            s_prime = np.stack(data[:, 3])  # s_t+1
            done = np.stack(data[:, 4])[:, None]  # done signal  (batchsize,1)

            if self._config["use_target_net"]:
                v_prime = self.Q_target.maxQ(s_prime)
            else:
                v_prime = self.Q.maxQ(s_prime)
            # target
            gamma = self._config["discount"]
            td_target = rew + gamma * (1.0 - done) * v_prime

            # optimize the lsq objective
            fit_loss, elementwise_loss = self.Q.fit(s, a, td_target)

            losses.append(fit_loss)

        return losses


class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        layer_sizes = [self.input_size] + self.hidden_sizes
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(i, o) for i, o in zip(layer_sizes[:-1], layer_sizes[1:])]
        )
        self.activations = [torch.nn.Tanh() for l in self.layers]
        self.readout = torch.nn.Linear(self.hidden_sizes[-1], self.output_size)

    def forward(self, x):
        for layer, activation_fun in zip(self.layers, self.activations):
            x = activation_fun(layer(x))
        return self.readout(x)

    def predict(self, x):
        with torch.no_grad():
            return self.forward(torch.from_numpy(x.astype(np.float32))).numpy()


class QFunction(Feedforward):
    def __init__(
        self, observation_dim, action_dim, hidden_sizes=[100, 100], learning_rate=0.0002
    ):
        super().__init__(
            input_size=observation_dim,
            hidden_sizes=hidden_sizes,
            output_size=action_dim,
        )
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate, eps=0.000001
        )
        self.loss = torch.nn.SmoothL1Loss(reduce=False)  # SmoothLoss

    def fit(self, observations, actions, targets):
        self.train()  # put model in training mode
        self.optimizer.zero_grad()
        # Forward pass
        acts = torch.from_numpy(actions)
        pred = self.Q_value(torch.from_numpy(observations).float(), acts)
        # Compute Loss
        elementwise_loss = self.loss(pred, torch.from_numpy(targets).float())
        loss = torch.mean(elementwise_loss)
        # Backward pass
        loss.backward()
        self.optimizer.step()
        return loss.item(), elementwise_loss.detach().numpy()

    def Q_value(self, observations, actions):
        return self.forward(observations).gather(1, actions[:, None])

    def maxQ(self, observations):
        return np.max(self.predict(observations), axis=-1, keepdims=True)

    def greedyAction(self, observations):
        return np.argmax(self.predict(observations), axis=-1)


class PrioritizedReplayBuffer:
    """Prioritized Replay buffer. Adapted to be similar to Memory from the lecture

    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight

    """

    def __init__(
        self, obs_dim: int, max_size: int, batch_size: int = 32, alpha: float = 0.6
    ):
        """Initialization."""
        assert alpha >= 0

        self.transitions = np.asarray([])
        self.size = 0
        self.current_idx = 0
        self.max_size = max_size
        
        
        # PER Stuff
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        self.batch_size = batch_size

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        # SegmentTree
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(self, transitions_new):
        if self.size == 0:
            blank_buffer = [np.asarray(transitions_new, dtype=object)] * self.max_size
            self.transitions = np.asarray(blank_buffer)

        self.transitions[self.current_idx, :] = np.asarray(
            transitions_new, dtype=object
        )
        self.size = min(self.size + 1, self.max_size)
        self.current_idx = (self.current_idx + 1) % self.max_size

        self.sum_tree[self.tree_ptr] = self.max_priority**self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority**self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_size

    def sample(self, batch=128, beta: float = 0.4):
        """Sample a batch of experiences."""
        self.batch_size = batch
        assert self.size >= self.batch_size
        assert beta > 0

        indices = self._sample_proportional()

        logging.debug(f"Transitions shape: {self.transitions.shape}")
        logging.debug(f"Transitions type: {type(self.transitions)}")
        logging.debug(f"Indices type: {type(indices)}")
        logging.debug(f"indices: {indices}")
        logging.debug(f"Number of indices: {len(indices)}")
        weights = np.array([self._calculate_weight(i, beta) for i in indices])
        return self.transitions[indices], indices, weights

    def update_priorities(self, indices, priorities):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < self.size

            self.sum_tree[idx] = priority**self.alpha
            self.min_tree[idx] = priority**self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self):
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, self.size - 1)
        segment = p_total / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        return np.array(indices, dtype=int)

    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * self.size) ** (-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * self.size) ** (-beta)
        weight = weight / max_weight

        return weight

    def get_all_transitions(self):
        return self.transitions[0 : self.size]


class QFunctionPrio(Feedforward):
    def __init__(
        self, observation_dim, action_dim, hidden_sizes=[100, 100], learning_rate=0.0002
    ):
        super().__init__(
            input_size=observation_dim,
            hidden_sizes=hidden_sizes,
            output_size=action_dim,
        )
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate, eps=0.000001
        )
        self.loss = torch.nn.SmoothL1Loss(reduction="none")  # MSELoss()

    def fit(self, observations, actions, targets, weights):
        self.train()  # put model in training mode
        self.optimizer.zero_grad()
        # Forward pass
        acts = torch.from_numpy(actions)
        pred = self.Q_value(torch.from_numpy(observations).float(), acts)
        # Compute Loss
        elementwise_loss = self.loss(pred, torch.from_numpy(targets).float())
        loss = (elementwise_loss * torch.from_numpy(weights).float()).mean()
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10)
        self.optimizer.step()
        return loss.item(), elementwise_loss.detach().numpy()

    def Q_value(self, observations, actions):
        return self.forward(observations).gather(1, actions[:, None])

    def maxQ(self, observations):
        return np.max(self.predict(observations), axis=-1, keepdims=True)

    def greedyAction(self, observations):
        return np.argmax(self.predict(observations), axis=-1)


class DQN_AGENT_priotized_buffer(object):
    """
    Agent implementing Q-learning with NN function approximation.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        alpha=0.2,
        beta=0.6,
        max_size=100000,
        config={
            "eps": 0.05,  # Epsilon in epsilon greedy policies
            "discount": 0.95,
            "buffer_size": int(1e5),
            "batch_size": 64,
            "learning_rate": 0.0002,
            "update_target_every": 20,
            "use_target_net": True,
            "prioritized_replay_eps": 1e-6,
        },
        **userconfig,
    ):

        if not isinstance(observation_space, spaces.box.Box):
            raise UnsupportedSpace(
                "Observation space {} incompatible "
                "with {}. (Require: Box)".format(observation_space, self)
            )
        if not isinstance(action_space, spaces.discrete.Discrete):
            raise UnsupportedSpace(
                "Action space {} incompatible with {}."
                " (Reqire Discrete.)".format(action_space, self)
            )

        self._observation_space = observation_space
        self._action_space = action_space
        self._action_n = action_space.n
        self._config = {
            "eps": 0.05,  # Epsilon in epsilon greedy policies
            "discount": 0.95,
            "buffer_size": int(1e5),
            "batch_size": 64,
            "learning_rate": 0.0002,
            "update_target_every": 20,
            "use_target_net": True,
            "prioritized_replay_eps": 1e-6,
        }
        self._config.update(userconfig)
        self.buffer = PrioritizedReplayBuffer(
            obs_dim=observation_space.shape[0],
            max_size=max_size,
            alpha=alpha,
            batch_size=self._config["batch_size"],
        )
        self._eps = self._config["eps"]
        self.priority_eps = self._config["prioritized_replay_eps"]

        self.alpha = alpha
        self.beta = beta
        self.sample_times = []
        # Q Network
        self.Q = QFunctionPrio(
            observation_dim=self._observation_space.shape[0],
            action_dim=self._action_n,
            learning_rate=self._config["learning_rate"],
        )
        # Q Network
        self.Q_target = QFunctionPrio(
            observation_dim=self._observation_space.shape[0],
            action_dim=self._action_n,
            learning_rate=0,
        )
        self._update_target_net()
        self.train_iter = 0

    def _update_target_net(self):
        self.Q_target.load_state_dict(self.Q.state_dict())

    def act(self, observation, eps=None):
        if eps is None:
            eps = self._eps
        # epsilon greedy
        if np.random.random() > eps:
            action = self.Q.greedyAction(observation)
        else:
            action = self._action_space.sample()
        return action

    def store_transition(self, transition):
        self.buffer.store(transition)

    def fit(self, iter_fit=32):

        losses = []
        self.train_iter += 1
        if (
            self._config["use_target_net"]
            and self.train_iter % self._config["update_target_every"] == 0
        ):
            self._update_target_net()
        for i in range(iter_fit):

            # sample from the replay buffer
            if self.buffer.size > self._config["batch_size"]:
                start_time = time.time()
                data, indices, weights = self.buffer.sample(
                    batch=self._config["batch_size"], beta=self.beta
                )
                logging.debug(f" Sample time: {time.time()- start_time}")
                self.sample_times.append(time.time()- start_time)
                s = np.stack(data[:, 0])  # s_t
                a = np.stack(data[:, 1])  # a_t
                rew = np.stack(data[:, 2])[:, None]  # rew  (batchsize,1)
                s_prime = np.stack(data[:, 3])  # s_t+1
                done = np.stack(data[:, 4])[:, None]  # done signal  (batchsize,1)

                if self._config["use_target_net"]:
                    v_prime = self.Q_target.maxQ(s_prime)
                else:
                    v_prime = self.Q.maxQ(s_prime)
                # target
                gamma = self._config["discount"]
                td_target = rew + gamma * (1.0 - done) * v_prime

                # optimize the lsq objective
                start_time = time.time()
                fit_loss, elementwise_loss = self.Q.fit(s, a, td_target, weights)
                logging.debug(f" Fit time: {time.time()- start_time}")
                priorities = elementwise_loss + self.priority_eps
                start_time = time.time()
                self.buffer.update_priorities(indices, priorities)
                logging.debug(f" Update time: {time.time()- start_time}")
                losses.append(fit_loss)
            else:
                break

        return losses
    
    def train(self, max_episodes=500, max_steps=500, fit_iterations=32, env=None):
        losses = []
        stats = []  
        frame_idx = 0
        
        # Beta annealing parameters
        beta_start = self.beta
        beta_frames = max_episodes * 700  # Increase this to make annealing slower
        
        time_start = time.time()
        for i in range(max_episodes):
            total_reward = 0
            ob, _info = env.reset()
            
            for t in range(max_steps):
                frame_idx += 1
                
                # Smoother beta annealing
                self.beta = min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)
                
                done = False        
                a = self.act(ob)
                start_time = time.time()
                (ob_new, reward, done, trunc, _info) = env.step(a)
                logging.debug(f" Env time: {time.time()- start_time}")
                total_reward += reward
                self.store_transition((ob, a, reward, ob_new, done))            
                ob = ob_new        
                if done: break    
                
            losses.extend(self.fit(fit_iterations))
            stats.append([i, total_reward, t+1])    
            
            if ((i-1)%20==0):
                print(f"{i}: Done after {t+1} steps. Reward: {total_reward} beta: {self.beta} frames: {frame_idx}")
        
        logging.debug(f" time per frame: {(time.time()-time_start)/frame_idx}")
        logging.debug(f" mean sample time: {np.mean(self.sample_times)}")
        return stats, losses
