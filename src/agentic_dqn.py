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
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# class to store transitions
class Memory:
    """Simple replay buffer as was given in the lecture
    transitions have the shape : (ob, a, reward, ob_new, done)"""

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
            # print(f" Sample time: {time.time()- start_time}")
            self.sample_times.append(time.time() - start_time)
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
        acts = torch.from_numpy(actions, device=device)
        pred = self.Q_value(torch.from_numpy(observations).float(), acts, device=device)
        # Compute Loss
        elementwise_loss = self.loss(
            pred, torch.from_numpy(targets).float(), device=device
        )
        loss = torch.mean(elementwise_loss, device=device)
        # Backward pass
        loss.backward()
        self.optimizer.step()
        return loss.item().cpu(), elementwise_loss.detach().numpy().cpu()

    def Q_value(self, observations, actions):
        return self.forward(observations).gather(1, actions[:, None])

    def maxQ(self, observations):
        return np.max(self.predict(observations), axis=-1, keepdims=True)

    def greedyAction(self, observations):
        return np.argmax(self.predict(observations), axis=-1)


class PrioritizedReplayBuffer:
    """Prioritized Replay buffer. Adapted to be similar to Memory from the lecture
     transitions have the shape : (ob, a, reward, ob_new, done)

    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight

    """

    def __init__(
        self,
        obs_dim: int,
        max_size: int,
        batch_size: int = 32,
        alpha: float = 0.6,
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
            # Transitions shape:  (array([-0.56146526, -0.8275003 , -0.9523803 ], dtype=float32), 3, -4.787382871690309, array([-0.6188719 , -0.78549194, -1.4230055 ], dtype=float32), False)

            print("Creating new buffer with size: ", self.max_size)
            print("Transitions shape: ", len(transitions_new))
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
        logging.debug(f"Indices shape: {indices.shape}")
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
        logging.debug(f"p_sample: {p_sample}")
        logging.debug(f"self.size: {self.size}")
        weight = (p_sample * self.size) ** (-beta)
        weight = weight / max_weight

        return weight

    def get_all_transitions(self):
        return self.transitions[0 : self.size]


class N_Step_ReplayBuffer:
    """N-step replay buffer. Adapted to be similar to Memory from the lecture can be used in Combination with PER

    transitions have the shape : (ob, a, reward, ob_new, done)"""

    def __init__(self, obs_dim, max_size=100000, n_steps=1, discount=0.95):
        self.transitions = np.asarray([])
        self.size = 0
        self.current_idx = 0
        self.max_size = max_size
        self.n_steps = n_steps
        self.discount = discount
        self.n_step_buffer = deque(maxlen=n_steps)

    def add_transition(self, transitions_new):
        if self.size == 0:
            blank_buffer = [np.asarray(transitions_new, dtype=object)] * self.max_size
            self.transitions = np.asarray(blank_buffer)

        self.n_step_buffer.append(transitions_new)

        if len(self.n_step_buffer) < self.n_steps:
            return transitions_new  # not enough transitions to add n-step transition

        # add n-step transition
        observation, action = self.n_step_buffer[0][
            :2
        ]  # get the first action and observation from the buffer
        reward, next_observation, done = self._get_n_step_info(self.current_idx)

        self.transitions[self.current_idx, :] = (
            np.asarray(  # add the n-step transition to the buffer
                [observation, action, reward, next_observation, done], dtype=object
            )
        )
        self.size = min(self.size + 1, self.max_size)
        self.current_idx = (self.current_idx + 1) % self.max_size
        return self.n_step_buffer[0]

    def sample(self, batch=1):
        if batch > self.size:
            batch = self.size
        self.inds = np.random.choice(range(self.size), size=batch, replace=False)
        return self.transitions[self.inds, :]

    def sample_from_idx(self, idx, batch=1):
        assert (
            len(idx) == batch
        ), f"Batch size {batch} does not match the length of the index in sample_from_idx {len(idx)}"

        assert (
            batch <= self.size
        ), "Batch size is larger than the buffer size in sample_from_idx"

        return self.transitions[idx, :]

    def _get_n_step_info(self, idx):
        rew, next_obs, done = self.n_step_buffer[-1][
            -3:
        ]  # take the last transitions out of the n_step_buffer

        # now we go back in the buffer and add the rewards
        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]

            rew = r + self.discount * rew * (1 - d)
            next_obs, done = (n_o, d) if d else (next_obs, done)

        return rew, next_obs, done


class QFunctionPrio(Feedforward):
    """Q function with prioritized replay buffer, this is the same as QFunction but with the elementwise loss  and some gpu operations"""

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

    def fit(
        self,
        observations,
        actions,
        targets,
        weights,
        n_step_obs=None,
        n_step_act=None,
        n_step_targets=None,
    ):
        self.train()  # put model in training mode
        self.optimizer.zero_grad()
        # Forward pass
        acts = torch.from_numpy(actions).to(device)
        pred = self.Q_value(torch.from_numpy(observations).float(), acts).to(device)
        # Compute Loss
        elementwise_loss = self.loss(pred, torch.from_numpy(targets).float().to(device))

        if n_step_obs is not None:
            n_acts = torch.from_numpy(n_step_act).to(device)
            n_pred = self.Q_value(torch.from_numpy(n_step_obs).float(), n_acts).to(
                device
            )
            n_elementwise_loss = self.loss(
                n_pred, torch.from_numpy(n_step_targets).float().to(device)
            )
            elementwise_loss = elementwise_loss + n_elementwise_loss
        # Backward pass
        loss = (elementwise_loss * torch.from_numpy(weights).float()).mean()
        loss.backward()
        # implement gradient clipping , seems to stabilize the training
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
        n_steps=1,
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
        self.n_steps = n_steps
        self.use_n_step = n_steps > 1
        self.buffer = PrioritizedReplayBuffer(
            obs_dim=observation_space.shape[0],
            max_size=max_size,
            alpha=alpha,
            batch_size=self._config["batch_size"],
        )
        if self.use_n_step:
            # we later combine the losses of the n-step transitions with the elementwise loss
            self.n_buffer = N_Step_ReplayBuffer(
                obs_dim=observation_space.shape[0],
                max_size=max_size,
                n_steps=n_steps,
                discount=config["discount"],
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
        ).to(device)
        # Q Network
        self.Q_target = QFunctionPrio(
            observation_dim=self._observation_space.shape[0],
            action_dim=self._action_n,
            learning_rate=0,
        ).to(device)
        logging.info(f"Using device: {device}")
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
                self.sample_times.append(time.time() - start_time)
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
                if self.use_n_step:
                    n_step_data = self.n_buffer.sample_from_idx(
                        indices, self._config["batch_size"]
                    )
                    n_s = np.stack(n_step_data[:, 0])

                    logging.debug(f" n_s shape: {n_s.shape}")

                    n_a = np.stack(n_step_data[:, 1])
                    n_rew = np.stack(n_step_data[:, 2])[:, None]
                    n_s_prime = np.stack(n_step_data[:, 3])

                    if self._config["use_target_net"]:
                        n_v_prime = self.Q_target.maxQ(n_s_prime)
                    else:
                        n_v_prime = self.Q.maxQ(n_s_prime)
                    n_done = np.stack(n_step_data[:, 4])[:, None]
                    n_td_target = n_rew + gamma * (1.0 - n_done) * n_v_prime

                    fit_loss, elementwise_loss = self.Q.fit(
                        s,
                        a,
                        td_target,
                        weights,
                        n_step_obs=n_s,
                        n_step_act=n_a,
                        n_step_targets=n_td_target,
                    )
                else:
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
                self.beta = min(
                    1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames
                )

                done = False
                a = self.act(ob)
                start_time = time.time()
                (ob_new, reward, done, trunc, _info) = env.step(a)
                logging.debug(f" Env time: {time.time()- start_time}")
                total_reward += reward

                if self.use_n_step:
                    one_step_transition = self.n_buffer.add_transition(
                        (ob, a, reward, ob_new, done)
                    )
                else:
                    one_step_transition = (ob, a, reward, ob_new, done)

                if one_step_transition is not ():
                    self.store_transition(one_step_transition)

                ob = ob_new
                if done:
                    break

            losses.extend(self.fit(fit_iterations))
            stats.append([i, total_reward, t + 1])

            if (i - 1) % 20 == 0:
                print(
                    f"{i}: Done after {t+1} steps. Reward: {total_reward} beta: {self.beta} frames: {frame_idx}"
                )

        logging.debug(f" time per frame: {(time.time()-time_start)/frame_idx}")
        logging.debug(f" mean sample time: {np.mean(self.sample_times)}")
        return stats, losses
