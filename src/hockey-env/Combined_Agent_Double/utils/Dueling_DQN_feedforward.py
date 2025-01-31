import torch
import numpy as np

class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, value_hidden_sizes = None, advantage_hidden_sizes = None):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_sizes  = hidden_sizes
        self.output_size  = output_size
        layer_sizes = [self.input_size] + self.hidden_sizes
        self.layers = torch.nn.ModuleList([ torch.nn.Linear(i, o) for i,o in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.activations = [ torch.nn.Tanh() for l in  self.layers ]
        #self.readout = torch.nn.Linear(self.hidden_sizes[-1], self.output_size)

        self.value_stream = torch.nn.Linear(hidden_sizes[-1], 1)
        self.advantage_stream = torch.nn.Linear(hidden_sizes[-1], output_size)      # (hidden_sizes[-1], action_dim)

        print("Dueling DQN Network Architecture:")
        print(f"  Input Layer: ({self.input_size})")
        for idx, (layer_in, layer_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            print(f"  Hidden Layer {idx + 1}: Linear({layer_in}, {layer_out}) -> Tanh")
        print(f"  Value Stream: Linear({hidden_sizes[-1]}, 1)")
        print(f"  Advantage Stream: Linear({hidden_sizes[-1]}, {output_size})")
        print("------------------------------------------------------")

    def forward(self, x):

        for layer, activation_fun in zip(self.layers, self.activations):
            x = activation_fun(layer(x))

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        return value + (advantage - advantage.mean(dim=-1, keepdim=True))

    def predict(self, x):
        with torch.no_grad():
            return self.forward(torch.from_numpy(x.astype(np.float32))).numpy()