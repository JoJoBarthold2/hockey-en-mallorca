import uuid
import torch
import random
import numpy as np
from comprl.client import Agent
import Agents.utils.memory_CUDA as mem
from Agents.utils.actions import MORE_ACTIONS
from Agents.Pablo.DQN_CUDA.QFunction import QFunction

class DQN(Agent):

    """Agent implementing DQN with CUDA."""

    def __init__(self, state_space, action_space, env = None, **userconfig):

        self.env = env

        self._state_space = state_space
        self._action_space = action_space
        self._action_n = action_space.n

        self.train_iter = 0

        self._config = {
            "eps": 0.05,
            "discount": 0.95,
            "buffer_size": int(1e5),
            "batch_size": 64,  # Before, 256 default
            "learning_rate": 0.0002,
            "update_target_every": 5,
            "use_target_net": True,
            "use_eps_decay": False,
            "eps_decay_mode": "exponential",
            "eps_min": 0.01,
            "eps_decay": 0.995,
            "seed": int(random.random()),
            "hidden_sizes": [128, 128],
            "use_more_actions": True,
        }

        self._config.update(userconfig)

        random.seed(self._config["seed"])
        np.random.seed(self._config["seed"])
        torch.manual_seed(self._config["seed"])

        # Ensure deterministic behavior in CUDA if available
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.cuda.manual_seed_all(self._config["seed"])
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        else:
            self.device = torch.device("cpu")

        print(f"Using {self.device}")

        self._eps = self._config["eps"]

        self.use_more_actions = self._config["use_more_actions"]

        self.buffer = mem.Memory(max_size = self._config["buffer_size"])

        # Q Network
        self.Q = QFunction(
            state_dim = self._state_space.shape[0],
            action_dim = self._action_n,
            learning_rate = self._config["learning_rate"],
            hidden_sizes = self._config["hidden_sizes"],
            device = self.device
        ).to(self.device)

        # Q Target
        self.Q_target = QFunction(
            state_dim = self._state_space.shape[0],
            action_dim = self._action_n,
            learning_rate = 0,  # We do not want to train the Target Function, only copy the weights of the Q Network
            hidden_sizes = self._config["hidden_sizes"],
            device = self.device
        ).to(self.device)
        self._update_target_net()

    def get_step(self, state):

        state = np.array(state)
        action = self.Q.greedyAction(state).tolist()

        if self.use_more_actions:
            continous_action = MORE_ACTIONS[action]
        else:
            continous_action = self.env.discrete_to_continous_action(action)

        return list(map(float, continous_action))

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(f"Game ended: {text_result} with my score: {stats[0]} against the opponent with score: {stats[1]}")

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int = int.from_bytes(game_id, byteorder = "little"))
        print(f"Game started (id: {game_id})")

    def _update_target_net(self):
        self.Q_target.load_state_dict(self.Q.state_dict())

    def act(self, state, eps = None):
        
        #print("ACT - State device:", state.device)  # Should be cuda:0

        eps = self._eps if eps is None else eps

        if np.random.random() > eps:
            action = self.Q.greedyAction(state)
        else: 
            action = self._action_space.sample()        
        return int(action.item())

    def _perform_epsilon_decay(self):

        if self._config["eps_decay_mode"] == "linear":
            self._eps = max(self._config["eps_min"], self._eps - self._config["eps_decay"])
        elif self._config["eps_decay_mode"] == "exponential":
            self._eps = max(self._config["eps_min"], self._eps * self._config["eps_decay"])
        else:
            raise ValueError("Error: Epsilon decay mode must be \"linear\" or \"exponential\".")

    def train(self, iter_fit = 32):

        losses = []
        self.train_iter += 1

        for i in range(iter_fit):
            
            if self.buffer.size > self._config["batch_size"]:

                s, a, rew, s_prime, done = self.buffer.sample(batch=self._config["batch_size"])

                if self._config["use_target_net"]:
                    v_prime = self.Q_target.maxQ(s_prime)
                else:
                    v_prime = self.Q.maxQ(s_prime)

                rew = rew.view(-1, 1)  # Convert to [batch_size, 1]
                done = done.view(-1, 1)  # Convert to [batch_size, 1]
                v_prime = v_prime.view(-1, 1)

                #print(f"Shape of v_prime: {v_prime.shape}")  # Debugging output

                # Target                                              
                td_target = rew + self._config["discount"] * (1.0 - done) * v_prime
                td_target = td_target.view(-1, 1)  # Ensure targets are [batch_size, 1]
                
                #print(f"Shape of rew: {rew.shape}, Shape of done: {done.shape}, Shape of td_target: {td_target.shape}")

                #print(f"Batch size used: {self._config['batch_size']}, Shape of s: {s.shape}")

                """print("TRAIN - Sampled States device:", s.device)
                print("TRAIN - Sampled Actions device:", a.device)
                print("TRAIN - Sampled Rewards device:", rew.device)
                print("TRAIN - Sampled Next States device:", s_prime.device)
                print("TRAIN - Sampled Dones device:", done.device)"""

                # optimize the lsq objective
                fit_loss = self.Q.fit(s, a, td_target)
                
                losses.append(fit_loss)
            else:
                break

        if self._config["use_target_net"] and self.train_iter % self._config["update_target_every"] == 0:
            self._update_target_net()

        return losses
