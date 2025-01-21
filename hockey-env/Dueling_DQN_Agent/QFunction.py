import numpy as np
import torch
import os

from Dueling_DQN_Agent.utils.Dueling_DQN_feedforward import Feedforward

class QFunction(Feedforward):

    def __init__(self, state_dim, action_dim, hidden_sizes=[128,128], learning_rate = 0.0002):

        super().__init__(input_size=state_dim, hidden_sizes=hidden_sizes, output_size=action_dim)

        self.optimizer=torch.optim.Adam(self.parameters(), 
                                        lr=learning_rate, 
                                        eps=0.000001)
        self.loss = torch.nn.SmoothL1Loss() # MSELoss()

    def fit(self, states, actions, targets):

        self.train() # put model in training mode
        self.optimizer.zero_grad()

        # Forward pass
        acts = torch.from_numpy(actions)
        pred = self.Q_value(torch.from_numpy(states).float(), acts)

        # Compute Loss
        loss = self.loss(pred, torch.from_numpy(targets).float())
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def Q_value(self, states, actions):

        """print("States type:", type(states), "Shape:", states.shape)
        print("Actions type:", type(actions), "Shape:", actions.shape, "Dtype:", actions.dtype)

        # Mostrar el contenido de la primera fila de cada variable
        print("First state:", states[0].tolist())  # Convertir a lista para mejor legibilidad
        print("First action:", actions[0].tolist())

        print("Output shape from forward:", self.forward(states).shape)"""

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