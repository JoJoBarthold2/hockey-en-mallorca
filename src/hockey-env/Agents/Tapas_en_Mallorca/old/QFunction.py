import numpy as np
import torch
import os

from Agents.Tapas_en_Mallorca.old.Feedforward import Feedforward

class QFunction(Feedforward):
    
    def __init__(self, state_dim, action_dim, use_dueling, hidden_sizes = [128,128], learning_rate = 0.0002, use_noisy = True, use_prio = True):

        super().__init__(input_size = state_dim, hidden_sizes = hidden_sizes, output_size = action_dim, use_dueling = use_dueling, use_noisy = use_noisy)

        self.use_noisy = use_noisy
        self.use_prio = use_prio

        self.optimizer=torch.optim.Adam(self.parameters(), 
                                        lr=learning_rate, 
                                        eps=0.000001)
        if self.use_prio:
            self.loss = torch.nn.SmoothL1Loss(reduction="none")
        else:
            self.loss = torch.nn.SmoothL1Loss()

    def fit(self, states, actions, targets, weights = None, n_step_obs = None, n_step_act = None, n_step_targets = None):

        self.train() # put model in training mode
        self.optimizer.zero_grad()

        # Forward pass
        acts = torch.from_numpy(actions)
        pred = self.Q_value(torch.from_numpy(states).float(), acts)

        # Compute Loss
        elementwise_loss = self.loss(pred, torch.from_numpy(targets).float())
        
        if n_step_obs is not None:
            n_acts = torch.from_numpy(n_step_act)
            n_pred = self.Q_value(torch.from_numpy(n_step_obs).float(), n_acts)

            n_elementwise_loss = self.loss(
                n_pred, torch.from_numpy(n_step_targets).float()
            )
            elementwise_loss = elementwise_loss + n_elementwise_loss

        # Backward pass

        if weights is not None:
            loss = (elementwise_loss * torch.from_numpy(weights).float()).mean()
        else:
            loss = elementwise_loss

        loss.backward()

        """torch.nn.utils.clip_grad_norm_(
            self.parameters(), max_norm=10
        )  # Implement gradient clipping seems to stabilize the training"""     # Not necesary since we are using ReLU

        self.optimizer.step()

        if weights is not None:
            return loss.item(), elementwise_loss.detach().numpy()
        else:
            return loss.item(), None

    def Q_value(self, states, actions):
        return self.forward(states).gather(1, actions[:,None])
    
    def maxQ(self, states):
        return np.max(self.predict(states), axis = -1, keepdims = True)

    def greedyAction(self, state):
        with torch.no_grad():
            action = np.argmax(self.forward(torch.tensor(state, dtype=torch.float32)).cpu().numpy(), axis=-1)
        return action

    def act(self, state):
        self.eval()
        with torch.no_grad():
            action = np.argmax(self.forward(torch.tensor(state, dtype = torch.float32)).cpu().numpy(), axis = -1)
        self.train()
        return action

    def save(self, env_name, name):
            os.makedirs(f"{env_name}/weights", exist_ok=True)
            torch.save(self.state_dict(), f"{env_name}/weights/{name}.pth")
            print(f"Network saved at {env_name}/weights/{name}.pth")

    def load(self, env_name, name = "training_finished"):
            self.load_state_dict(torch.load(f"{env_name}/weights/{name}.pth"))
            print(f"Network loaded from {env_name}/weights/{name}.pth")