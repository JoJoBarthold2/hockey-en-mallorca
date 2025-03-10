import torch
import numpy as np

class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_sizes  = hidden_sizes
        self.output_size  = output_size
        layer_sizes = [self.input_size] + self.hidden_sizes
        self.layers = torch.nn.ModuleList([ torch.nn.Linear(i, o) for i,o in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.activations = [ torch.nn.Tanh() for l in  self.layers ]
        self.value_stream = torch.nn.Linear(hidden_sizes[-1], 1)
        self.advantage_stream = torch.nn.Linear(hidden_sizes[-1], output_size)

        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)
        
        torch.nn.init.xavier_normal_(self.value_stream.weight)
        torch.nn.init.zeros_(self.value_stream.bias)

        torch.nn.init.xavier_normal_(self.advantage_stream.weight)
        torch.nn.init.zeros_(self.advantage_stream.bias)

    def forward(self, x):
        for layer, activation_fun in zip(self.layers, self.activations):
            x = activation_fun(layer(x))

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        return value + (advantage - advantage.mean(dim = -1, keepdim = True))

    def predict(self, x):
        with torch.no_grad():
            return self.forward(torch.from_numpy(x.astype(np.float32))).numpy()
