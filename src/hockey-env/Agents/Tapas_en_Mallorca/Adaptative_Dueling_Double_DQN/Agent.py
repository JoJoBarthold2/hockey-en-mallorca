import uuid
import torch
import random
import numpy as np
from comprl.client import Agent
import Agents.utils.memory as mem
import Agents.utils.prioritized_replay_buffer as prio_mem 
import Agents.utils.n_step_replay_buffer as n_step_mem
from Agents.utils.actions import MORE_ACTIONS
from Agents.Tapas_en_Mallorca.Adaptative_Dueling_Double_DQN.QFunction import QFunction


class Adaptative_Dueling_Double_DQN_better_mem(Agent):

    """Agent implementing Adaptative Dueling Double DQN."""

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
            "batch_size": 64,  
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
            "use_dueling": True,
            "use_double": True,
            "use_noisy": False,
             "alpha": 0.2,
            "beta": 0.6,
            "max_size": 100000,
            "n_steps": 4,   # Use n-step by default
            "use_prio": True,
            "prioritized_replay_eps": 1e-6,
        }

        self._config.update(userconfig)
        self.n_gamma = self._config["discount"] ** self._config["n_steps"]
        random.seed(self._config["seed"])
        np.random.seed(self._config["seed"])
        torch.manual_seed(self._config["seed"])

        # Ensure deterministic behavior in CUDA if available
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._config["seed"])
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self._eps = self._config["eps"]

        self.use_more_actions = self._config["use_more_actions"]

        self.use_dueling = self._config["use_dueling"]
        self.use_double = self._config["use_double"]
        self.use_noisy = self._config["use_noisy"]


        #Memory 
        self.use_prio = self._config["use_prio"]
          
        if self._config["use_prio"]:
            self.buffer = prio_mem.PrioritizedReplayBuffer(
                max_size = self._config["max_size"],
                batch_size = self._config["batch_size"],    
                alpha = self._config["alpha"],
            )
           
        else:
            self.buffer = mem.Memory(
                max_size = self._config["max_size"],
             
            )
        

        self.priority_eps = self._config["prioritized_replay_eps"]
        self.beta = self._config["beta"]
        self.alpha = self._config["alpha"]
        ## N-step
        self.n_step = self._config["n_steps"]
        if self.n_step > 1:
            # we later combine the losses of the n-step transitions with the elementwise loss
            self.n_buffer = n_step_mem.N_Step_ReplayBuffer(
                    obs_dim = self._state_space.shape[0],
                    max_size = self._config["max_size"],
                    n_steps = self.n_step,
                    discount = self._config["discount"]
                )
            self.use_n_step = True
        else:
            self.use_n_step = False


        # Q Network
        self.Q = QFunction(
            state_dim = self._state_space.shape[0],
            action_dim = self._action_n,
            learning_rate = self._config["learning_rate"],
            hidden_sizes = self._config["hidden_sizes"],
            use_dueling = self.use_dueling,
            use_noisy = self.use_noisy
        )

        # Q Target
        self.Q_target = QFunction(
            state_dim = self._state_space.shape[0],
            action_dim = self._action_n,
            learning_rate = 0,  # We do not want to train the Target Function, only copy the weights of the Q Network
            hidden_sizes = self._config["hidden_sizes"],
            use_dueling = self.use_dueling,
            use_noisy = self.use_noisy
        )
        self._update_target_net()

    def get_step(self, state):

        state = np.array(state)
           
        if self.use_noisy:
            self.Q.reset_noise()
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

    def act(self, state, eps = None, validation = False):
        
        if self.use_noisy:
            self.Q.reset_noise() 
            return self.Q.greedyAction(state)
        
        eps = self._eps if eps is None else eps

        if np.random.random() > eps:
            action = self.Q.greedyAction(state)
        else: 
            action = self._action_space.sample()        

        return action

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
                if self._config["use_prio"]:
                  
                    data, indices, weights = self.buffer.sample(batch = self._config["batch_size"], beta = self.beta)
                  
                else:   
                  
                    data = self.buffer.sample(batch = self._config["batch_size"])
                  
                s = np.stack(data[:, 0])                # Current state (s_t)
                a = np.stack(data[:, 1])                # Action taken (a_t)
                rew = np.stack(data[:, 2])[:, None]     # Reward received (r)
                s_prime = np.stack(data[:, 3])          # Next state (s_t+1)
                done = np.stack(data[:,4])[:,None]      # Done flag (1 if terminal, else 0)

                # Double DQN
                if self.use_double:
                    if self._config["use_target_net"]:
                        a_prime = self.Q.greedyAction(s_prime)     
                        s_prime_tensor = torch.tensor(s_prime, dtype=torch.float32)
                        a_prime_tensor = torch.tensor(a_prime, dtype=torch.int64)
                        v_prime = self.Q_target.Q_value(s_prime_tensor, a_prime_tensor)     # Evaluate it using Q_target
                    else:
                        a_prime = self.Q.greedyAction(s_prime)  
                        s_prime_tensor = torch.tensor(s_prime, dtype=torch.float32) # We need torch here for the Q_value function
                        a_prime_tensor = torch.tensor(a_prime, dtype=torch.int64)
                        v_prime = self.Q.Q_value(s_prime_tensor, a_prime_tensor)

                                                                
                    td_target = rew + self._config["discount"] * (1.0 - done) * v_prime.detach().numpy()
                else:
                    if self._config["use_target_net"]:
                        v_prime = self.Q_target.maxQ(s_prime)
                    else:
                        v_prime = self.Q.maxQ(s_prime)

                                                                
                    td_target = rew + self._config["discount"] * (1.0 - done) * v_prime
                
                if self.use_n_step:
                    if self.use_prio:
                        n_step_data = self.n_buffer.sample_from_idx(
                        indices, self._config["batch_size"]
                    )
                    else: 
                        n_step_data = self.n_buffer.sample(self._config["batch_size"])


                    n_s = np.stack(n_step_data[:, 0])
                    n_a = np.stack(n_step_data[:, 1])
                    n_rew = np.stack(n_step_data[:, 2])[:, None]
                    n_s_prime = np.stack(n_step_data[:, 3])
                    n_done = np.stack(n_step_data[:, 4])[:, None]

                    if self._config["use_double"]:     
                       
                        if self._config["use_target_net"]:
                            n_a_prime = self.Q.greedyAction(n_s_prime)  
                            n_s_prime_tensor = torch.tensor(n_s_prime, dtype = torch.float32)
                            n_a_prime_tensor = torch.tensor(n_a_prime, dtype = torch.int64)
                            n_v_prime = self.Q_target.Q_value(n_s_prime_tensor, n_a_prime_tensor)     # Evaluate it using Q_target
                        else:
                            n_a_prime = self.Q.greedyAction(n_s_prime)      
                            n_s_prime_tensor = torch.tensor(n_s_prime, dtype = torch.float32)
                            n_a_prime_tensor = torch.tensor(n_a_prime, dtype = torch.int64)
                            n_v_prime = self.Q.Q_value(n_s_prime_tensor, n_a_prime_tensor) 

                        # Target
                        n_td_target = (n_rew + self._config["discount"] * (1.0 - n_done) * n_v_prime.detach().numpy())
                        
                        

                    else:       # Without Double DQN but with N-step
                        if self._config["use_target_net"]:
                            n_v_prime = self.Q_target.maxQ(n_s_prime)
                        else:
                            n_v_prime = self.Q.maxQ(n_s_prime)

                       
                        n_td_target = n_rew + self.n_gamma * (1.0 - n_done) * n_v_prime

                    if self.use_prio:
                        fit_loss, elementwise_loss = self.Q.fit(
                            s,
                            a,
                            td_target,
                            weights,
                            n_step_obs = n_s,
                            n_step_act = n_a,
                            n_step_targets = n_td_target,
                        )
                        priorities = elementwise_loss + self.priority_eps
                        self.buffer.update_priorities(indices, priorities)
                    else:
                        fit_loss, _ = self.Q.fit(
                            s,
                            a,
                            td_target,
                            n_step_obs = n_s,
                            n_step_act = n_a,
                            n_step_targets = n_td_target,
                        )

                elif self.use_prio and not self.use_n_step:
                     
                        fit_loss, elementwise_loss = self.Q.fit(s, a, td_target, weights)
                    
                        priorities = elementwise_loss + self.priority_eps
                      
                        self.buffer.update_priorities(indices, priorities)
                   

                else:
                  
                    fit_loss, _ = self.Q.fit(s, a, td_target)
                 

                losses.append(fit_loss)
                
            else:
                break

        if self._config["use_target_net"] and self.train_iter % self._config["update_target_every"] == 0:
            self._update_target_net()

        return losses
