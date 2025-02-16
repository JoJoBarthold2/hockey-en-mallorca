import os
import time  # Only for debugging
import torch
import pickle
import random
import logging
import numpy as np
import uuid
from comprl.client import Agent

from Prio_n_step_Agent.QFunction import QFunction
import Prio_n_step_Agent.utils.n_step_replay_buffer as rb
import Prio_n_step_Agent.utils.prioritized_replay_buffer as mem
from Prio_n_step_Agent.utils.actions import MORE_ACTIONS


class Prio_DQN_Agent(Agent):
    """Agent implementing Q-learning with NN function approximation."""

    def __init__(
        self,
        state_space,
        action_space,
        alpha=0.2,
        beta=0.6,
        max_size=100000,
        n_steps=1,
        use_more_actions=True,
        env=None,
        **userconfig,
    ):

        self._state_space = state_space
        self._action_space = action_space
        self._action_n = action_space.n
        self.use_more_actions = use_more_actions
        self.env = env
        self.train_iter = 0

        self._config = {
            "eps": 0.05,
            "discount": 0.95,
            "buffer_size": int(1e5),
            "batch_size": 64,  # Before, Pablo had 256 default
            "learning_rate": 0.0002,
            "update_target_every": 20,
            "use_target_net": True,
            "use_eps_decay": False,
            "eps_decay_mode": "exponential",
            "eps_min": 0.01,
            "eps_decay": 0.995,
            "seed": int(random.random()),
            "hidden_sizes": [128, 128],
            "value_hidden_sizes": None,
            "advantage_hidden_sizes": None,
            "prioritized_replay_eps": 1e-6,
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

        self.n_steps = n_steps
        self.use_n_step = n_steps > 1

        self.buffer = mem.PrioritizedReplayBuffer(
            max_size=max_size,
            alpha=alpha,
            batch_size=self._config["batch_size"],
        )

        if self.use_n_step:
            # we later combine the losses of the n-step transitions with the elementwise loss
            self.n_buffer = rb.N_Step_ReplayBuffer(
                obs_dim=state_space.shape[0],
                max_size=max_size,
                n_steps=n_steps,
                discount=self._config["discount"],
            )

        self._eps = self._config["eps"]
        self.priority_eps = self._config["prioritized_replay_eps"]

        self.alpha = alpha
        self.beta = beta
        self.sample_times = []

        # Q Network
        self.Q = QFunction(
            state_dim=self._state_space.shape[0],
            action_dim=self._action_n,
            learning_rate=self._config["learning_rate"],
            hidden_sizes=self._config["hidden_sizes"],
            value_hidden_sizes=self._config["value_hidden_sizes"],
            advantage_hidden_sizes=self._config["advantage_hidden_sizes"],
        )

        # Q Target
        self.Q_target = QFunction(
            state_dim=self._state_space.shape[0],
            action_dim=self._action_n,
            learning_rate=0,  # We do not want to train the Target Function, only copy the weights of the Q Network
            hidden_sizes=self._config["hidden_sizes"],
            value_hidden_sizes=self._config["value_hidden_sizes"],
            advantage_hidden_sizes=self._config["advantage_hidden_sizes"],
        )  # We do not want to train the Target Function, only copy the weights of the Q Network
        self._update_target_net()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")

    def _update_target_net(self):
        self.Q_target.load_state_dict(self.Q.state_dict())

    def act(self, state):  # Fuction to be consistent with naming for self-play
        self.perform_greedy_action(state, eps=0)

    def get_step(self, state):
        state = np.array(state)

        #

        action = self.Q.greedyAction(state).tolist()
        if self.use_more_actions:
            continous_action = MORE_ACTIONS[action]
        else:
            continous_action = self.env.discrete_to_continous_action(action)
        action_list = list(map(float, continous_action))

        return action_list

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"Game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )

    def perform_greedy_action(self, state, eps=None):

        if eps is None:
            eps = self._eps

        if np.random.random() > eps:
            action = self.Q.greedyAction(state)
        else:
            action = self._action_space.sample()
        # print(f"action: {action}, shape: {np.asarray(action).shape}")
        return action

    def _perform_epsilon_decay(self):

        if self._config["eps_decay_mode"] == "linear":
            self._eps = max(
                self._config["eps_min"], self._eps - self._config["eps_decay"]
            )
        elif self._config["eps_decay_mode"] == "exponential":
            self._eps = max(
                self._config["eps_min"], self._eps * self._config["eps_decay"]
            )
        else:
            raise ValueError(
                'Error: Epsilon decay mode must be "linear" or "exponential".'
            )

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id))
        print(f"Game started (id: {game_id})")

    def train(self, iter_fit=32):

        losses = []
        self.train_iter += 1

        for i in range(iter_fit):

            if self.buffer.size > self._config["batch_size"]:

                # Sample from the replay buffer

                start_time = time.time()  # Debugging

                data, indices, weights = self.buffer.sample(
                    batch=self._config["batch_size"], beta=self.beta
                )

                logging.debug(f" Sample time: {time.time()- start_time}")  # Debugging
                self.sample_times.append(time.time() - start_time)

                s = np.stack(data[:, 0])  # Current state (s_t)
                a = np.stack(data[:, 1])  # Action taken (a_t)
                rew = np.stack(data[:, 2])[:, None]  # Reward received (r)
                s_prime = np.stack(data[:, 3])  # Next state (s_t+1)
                done = np.stack(data[:, 4])[
                    :, None
                ]  # Done flag (1 if terminal, else 0)

                if self._config["use_target_net"]:
                    v_prime = self.Q_target.maxQ(s_prime)
                else:
                    v_prime = self.Q.maxQ(s_prime)
                # Target

                td_target = rew + self._config["discount"] * (1.0 - done) * v_prime

                # Optimize the lsq objective

                start_time = time.time()  # Debugging

                if self.use_n_step:

                    n_step_data = self.n_buffer.sample_from_idx(
                        indices, self._config["batch_size"]
                    )
                    n_s = np.stack(n_step_data[:, 0])

                    logging.debug(f" n_s shape: {n_s.shape}")  # Debugging

                    n_a = np.stack(n_step_data[:, 1])
                    n_rew = np.stack(n_step_data[:, 2])[:, None]
                    n_s_prime = np.stack(n_step_data[:, 3])

                    if self._config["use_target_net"]:
                        n_v_prime = self.Q_target.maxQ(n_s_prime)
                    else:
                        n_v_prime = self.Q.maxQ(n_s_prime)  # Evaluate it using Q_target

                    n_done = np.stack(n_step_data[:, 4])[:, None]
                    n_td_target = (
                        n_rew + self._config["discount"] * (1.0 - n_done) * n_v_prime
                    )

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

                logging.debug(f" Fit time: {time.time()- start_time}")  # Debugging

                priorities = elementwise_loss + self.priority_eps

                start_time = time.time()  # Debugging

                self.buffer.update_priorities(indices, priorities)

                logging.debug(f" Update time: {time.time()- start_time}")  # Debugging

                losses.append(fit_loss)

            else:
                break

        if (
            self._config["use_target_net"]
            and self.train_iter % self._config["update_target_every"] == 0
        ):
            self._update_target_net()

        return losses
