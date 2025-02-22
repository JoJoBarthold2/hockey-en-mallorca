import torch
import numpy as np

# class to store transitions
class Memory():

    def __init__(self, max_size = 100000):

        self.transitions = np.asarray([])
        self.size = 0
        self.current_idx = 0
        self.max_size = max_size

    def store(self, transitions_new):
        if torch.cuda.is_available():
            states, actions, rewards, next_states, dones = transitions_new

            states = states.to("cuda") if isinstance(states, torch.Tensor) else torch.tensor(states, dtype=torch.float32, device="cuda")
            next_states = next_states.to("cuda") if isinstance(next_states, torch.Tensor) else torch.tensor(next_states, dtype=torch.float32, device="cuda")
            actions = actions.to("cuda") if isinstance(actions, torch.Tensor) else torch.tensor(actions, dtype=torch.long, device="cuda")
            rewards = rewards.to("cuda") if isinstance(rewards, torch.Tensor) else torch.tensor(rewards, dtype=torch.float32, device="cuda")
            dones = dones.to("cuda") if isinstance(dones, torch.Tensor) else torch.tensor(dones, dtype=torch.float32, device="cuda")

            transitions_new = (states, actions, rewards, next_states, dones)

            if self.size == 0:
                blank_buffer = [transitions_new] * self.max_size
                self.transitions = blank_buffer  
            self.transitions[self.current_idx] = transitions_new
            self.size = min(self.size + 1, self.max_size)
            self.current_idx = (self.current_idx + 1) % self.max_size

            """print("Store - State device:", states.device)
            print("Store - Next state device:", next_states.device)
            print("Store - Actions device:", actions.device)
            print("Store - Rewards device:", rewards.device)
            print("Store - Dones device:", dones.device)"""

    def sample(self, batch=1):
        if batch > self.size:
            batch = self.size
        self.inds = np.random.choice(range(self.size), size=batch, replace=False)

        if torch.cuda.is_available():
            batch_data = [self.transitions[i] for i in self.inds]
            states, actions, rewards, next_states, dones = zip(*batch_data)

            states = [s.clone().detach() if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32, device="cuda") for s in states]
            next_states = [s.clone().detach() if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32, device="cuda") for s in next_states]

            states = torch.stack(states).to("cuda")
            next_states = torch.stack(next_states).to("cuda")
            actions = torch.tensor(actions, dtype=torch.long, device="cuda")
            rewards = torch.tensor(rewards, dtype=torch.float32, device="cuda")
            dones = torch.tensor(dones, dtype=torch.float32, device="cuda")

            """print("Sample - State device:", states.device)
            print("Sample - Next state device:", next_states.device)
            print("Sample - Actions device:", actions.device)
            print("Sample - Rewards device:", rewards.device)
            print("Sample - Dones device:", dones.device)"""

            return states, actions, rewards, next_states, dones

    def get_all_transitions(self):
        return self.transitions[0:self.size]
