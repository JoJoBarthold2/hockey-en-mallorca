import torch
import numpy as np
from Agents.utils.noisy_linear_remake import NoisyLinear

class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_sizes  = hidden_sizes
        self.output_size  = output_size
        layer_sizes = [self.input_size] + self.hidden_sizes
        self.layers = torch.nn.ModuleList()
        
        """for i in range(len(layer_sizes) - 1):
            if self.use_noisy:
                self.layers.append(NoisyLinear(layer_sizes[i], layer_sizes[i + 1]))
            else:
                layer = torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1])
                torch.nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)
                self.layers.append(layer)"""
        
        self.layers.append(torch.nn.Linear(layer_sizes[0], layer_sizes[1]))
        torch.nn.init.xavier_normal_(self.layers[0].weight)
        if self.layers[0].bias is not None:
            torch.nn.init.zeros_(self.layers[0].bias)

        for i in range(1, len(layer_sizes) - 1):
            self.layers.append(NoisyLinear(layer_sizes[i], layer_sizes[i + 1]))

        self.activations = [torch.nn.Tanh() for _ in self.layers]
                
        # Output Layer (Readout)
        """if self.use_noisy:
            self.readout = NoisyLinear(self.hidden_sizes[-1], self.output_size)
        else:
            self.readout = torch.nn.Linear(self.hidden_sizes[-1], self.output_size)
            torch.nn.init.xavier_normal_(self.readout.weight)
            torch.nn.init.zeros_(self.readout.bias)"""
        self.readout = NoisyLinear(self.hidden_sizes[-1], self.output_size)

    def forward(self, x):
        for layer,activation_fun in zip(self.layers, self.activations):
            x = activation_fun(layer(x))
        return self.readout(x)

    def predict(self, x):
        with torch.no_grad():
            return self.forward(torch.from_numpy(x.astype(np.float32))).numpy()
        
    def reset_noise(self):
        """if self.use_noisy:
            for layer in self.layers:
                if isinstance(layer, NoisyLinear):
                    layer.reset_noise()
            self.readout.reset_noise()"""
        for layer in self.layers:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()
        self.readout.reset_noise()