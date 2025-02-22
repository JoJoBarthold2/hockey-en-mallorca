import torch
import os

from Agents.Pablo.DQN_CUDA.Feedforward import Feedforward

class QFunction(Feedforward):
    
    def __init__(self, state_dim, action_dim, device, hidden_sizes = [128,128], learning_rate = 0.0002):

        super().__init__(input_size = state_dim, hidden_sizes = hidden_sizes, output_size = action_dim)

        self.device = device

        self.optimizer=torch.optim.Adam(self.parameters(), 
                                        lr = learning_rate, 
                                        eps = 0.000001)
        self.loss = torch.nn.SmoothL1Loss()

    def fit(self, states, actions, targets):

        self.train() # put model in training mode
        self.optimizer.zero_grad()

        states = states.clone().detach() if isinstance(states, torch.Tensor) else torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = actions.clone().detach() if isinstance(actions, torch.Tensor) else torch.tensor(actions, dtype=torch.long, device=self.device)
        targets = targets.clone().detach() if isinstance(targets, torch.Tensor) else torch.tensor(targets, dtype=torch.float32, device=self.device)

        targets = targets.view(-1, 1)

        """print("FIT - States device:", states.device)  # Should be cuda:0
        print("FIT - Actions device:", actions.device)
        print("FIT - Targets device:", targets.device)"""

        # Forward pass
        """acts = torch.from_numpy(actions)"""
        pred = self.Q_value(states, actions)

        # Compute Loss
        loss = self.loss(pred, targets)
        
        """print("FIT - Predictions device:", pred.device)  # Should be cuda:0
        print("FIT - Loss device:", loss.device)  # Should be cuda:0"""

        # Backward pass
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def Q_value(self, states, actions):
        """print("Q_VALUE - States device:", states.device)
        print("Q_VALUE - Actions device:", actions.device)"""
        return self.forward(states).gather(1, actions[:,None])
    
    def maxQ(self, states):

        #print("MAXQ - States device:", states.device)
        
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, dtype = torch.float32, device = self.device)
        else:
            states = states.clone().detach()

        if states.dim() == 1:  # Ensure batch dimension
            states = states.unsqueeze(0)

        return self.forward(states).max(dim = 1, keepdim = True)[0]

    def greedyAction(self, states):

        #print(states)

        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, dtype = torch.float32, device = self.device)
        else:
            states = states.clone().detach()

        if states.dim() == 1:
            states = states.unsqueeze(0)

        #print("GREEDY ACTION - States device:", states.device)

        return self.forward(states).argmax(dim=1).squeeze(0)

    def save(self, env_name, name):
            os.makedirs(f"{env_name}/weights", exist_ok=True)
            torch.save(self.state_dict(), f"{env_name}/weights/{name}.pth")
            print(f"Network saved at {env_name}/weights/{name}.pth")

    def load(self, env_name, name = "training_finished"):
            self.load_state_dict(torch.load(f"{env_name}/weights/{name}.pth"))
            print(f"Network loaded from {env_name}/weights/{name}.pth")