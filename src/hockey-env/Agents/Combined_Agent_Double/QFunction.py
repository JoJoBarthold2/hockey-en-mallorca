import numpy as np
import torch
import os

from Agents.Combined_Agent_Double.utils.Dueling_DQN_feedforward import Feedforward

class QFunction(Feedforward):
    
    """ Q function with prioritized replay buffer, this is the same as QFunction but with the elementwise loss and some gpu operations. """

    def __init__(self, state_dim, action_dim, hidden_sizes = [128,128], value_hidden_sizes = None, advantage_hidden_sizes = None, learning_rate = 0.0002):

        super().__init__(input_size = state_dim, hidden_sizes = hidden_sizes, output_size = action_dim)

        self.optimizer=torch.optim.Adam(self.parameters(), 
                                        lr=learning_rate, 
                                        eps=0.000001)
        self.loss = torch.nn.SmoothL1Loss(reduction = "none")

    def fit(self, states, actions, targets, weights, n_step_obs = None, n_step_act = None, n_step_targets = None,):

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
        loss = (elementwise_loss * torch.from_numpy(weights).float()).mean()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10)      # Implement gradient clipping seems to stabilize the training

        self.optimizer.step()

        return loss.item(), elementwise_loss.detach().numpy()

    def Q_value(self, states, actions):
        return self.forward(states).gather(1, actions[:,None])
    
    def maxQ(self, states):
        return np.max(self.predict(states), axis=-1, keepdims=True)
        
    def greedyAction(self, states):
        return np.argmax(self.predict(states), axis=-1)
    
    def save(self, env_name, name):
            os.makedirs(f"{env_name}/weights", exist_ok=True)
            torch.save(self.state_dict(), f"{env_name}/weights/{name}.pth")
            print(f"Network saved at {env_name}/weights/{name}.pth")

    def load(self, env_name, name = "training_finished"):
            self.load_state_dict(torch.load(f"{env_name}/weights/{name}.pth"))
            print(f"Network loaded from {env_name}/weights/{name}.pth")