import torch
import numpy as np

class Feedforward(torch.nn.Module):

    def __init__(self, input_size, hidden_sizes, output_size, use_dueling):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.use_dueling = use_dueling

        layer_sizes = [self.input_size] + self.hidden_sizes
        self.layers = torch.nn.ModuleList([torch.nn.Linear(i, o) for i, o in zip(layer_sizes[:-1], layer_sizes[1:])])
        self.activations = [torch.nn.Tanh() for _ in self.layers]

        if self.use_dueling:
            # Dueling Architecture
            self.value_stream = torch.nn.Linear(hidden_sizes[-1], 1)
            self.advantage_stream = torch.nn.Linear(hidden_sizes[-1], output_size)
        else:
            # Standard Feedforward Network
            self.readout = torch.nn.Linear(self.hidden_sizes[-1], self.output_size)
        
        self._print_architecture(layer_sizes)
    
    def _print_architecture(self, layer_sizes):
        architecture_type = "Dueling Network Architecture" if self.use_dueling else "Standard Feedforward Architecture"
        print(architecture_type)
        print(f"  Input Layer: ({self.input_size})")
        activation_type = self.activations[0].__class__.__name__ if self.activations else "None"
        for idx, (layer_in, layer_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            print(f"  Hidden Layer {idx + 1}: Linear({layer_in}, {layer_out}) -> {activation_type}")
        if self.use_dueling:
            print(f"  Value Stream: Linear({self.hidden_sizes[-1]}, 1)")
            print(f"  Advantage Stream: Linear({self.hidden_sizes[-1]}, {self.output_size})")
        else:
            print(f"  Output Layer: Linear({self.hidden_sizes[-1]}, {self.output_size})")
        print("------------------------------------------------------")

    def forward(self, x):
        for layer, activation_fun in zip(self.layers, self.activations):
            x = activation_fun(layer(x))
        
        if self.use_dueling:
            value = self.value_stream(x)
            advantage = self.advantage_stream(x)
            return value + (advantage - advantage.mean(dim=-1, keepdim=True))
        else:
            return self.readout(x)

    def predict(self, x):
        with torch.no_grad():
            return self.forward(torch.from_numpy(x.astype(np.float32))).numpy()