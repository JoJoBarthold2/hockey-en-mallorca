import torch
import numpy as np
from Agents.utils.noisy_linear_remake import NoisyLinear

class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, use_dueling = True, use_noisy = False):
        
        super(Feedforward, self).__init__()

        self.use_dueling = use_dueling
        self.use_noisy = use_noisy
        self.input_size = input_size
        self.hidden_sizes  = hidden_sizes
        self.output_size  = output_size
        layer_sizes = [self.input_size] + self.hidden_sizes
        self.layers = torch.nn.ModuleList()

        self.layers.append(torch.nn.Linear(layer_sizes[0], layer_sizes[1]))     # First layer always linear
        torch.nn.init.xavier_normal_(self.layers[0].weight)
        if self.layers[0].bias is not None:
            torch.nn.init.zeros_(self.layers[0].bias)

        for i in range(1, len(layer_sizes) - 1):
            if self.use_noisy:
                self.layers.append(NoisyLinear(layer_sizes[i], layer_sizes[i + 1]))
            else:
                layer = torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1])
                torch.nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)
                self.layers.append(layer)

        self.activations = [torch.nn.Tanh() for _ in self.layers]

        if self.use_dueling:
            self.value_stream = torch.nn.Linear(hidden_sizes[-1], 1)
            self.advantage_stream = torch.nn.Linear(hidden_sizes[-1], output_size)
            torch.nn.init.xavier_normal_(self.value_stream.weight)
            torch.nn.init.zeros_(self.value_stream.bias)
            torch.nn.init.xavier_normal_(self.advantage_stream.weight)
            torch.nn.init.zeros_(self.advantage_stream.bias)
        else:
            if self.use_noisy:
                self.readout = NoisyLinear(self.hidden_sizes[-1], self.output_size)
            else:
                self.readout = torch.nn.Linear(self.hidden_sizes[-1], self.output_size)
                torch.nn.init.xavier_normal_(self.readout.weight)
                torch.nn.init.zeros_(self.readout.bias)

        self._print_architecture()
    
    def _print_architecture(self):

        architecture_type = "Dueling Network Architecture" if self.use_dueling else "Standard Feedforward Architecture"
        noise_type = " with Noisy Layers" if self.use_noisy else ""
        print(f"{architecture_type}{noise_type}")

        print(f"  Input Layer: ({self.input_size})")

        for idx, (layer, activation) in enumerate(zip(self.layers, self.activations)):
            layer_type = layer.__class__.__name__       # Dynamically detect layer type (Linear or NoisyLinear)
            activation_type = activation.__class__.__name__ if activation is not None else "None"
            in_features = layer.in_features if hasattr(layer, "in_features") else "?"
            out_features = layer.out_features if hasattr(layer, "out_features") else "?"
            
            print(f"  Hidden Layer {idx + 1}: {layer_type}({in_features}, {out_features}) -> {activation_type}")
        
        if self.use_dueling:
            print(f"  Value Stream: {self.value_stream.__class__.__name__}({self.hidden_sizes[-1]}, 1)")
            print(f"  Advantage Stream: {self.advantage_stream.__class__.__name__}({self.hidden_sizes[-1]}, {self.output_size})")
        else:
            print(f"  Output Layer: {self.readout.__class__.__name__}({self.hidden_sizes[-1]}, {self.output_size})")

        print("------------------------------------------------------")

    def forward(self, x):
        for layer, activation_fun in zip(self.layers, self.activations):
            x = activation_fun(layer(x))

        if self.use_dueling:
            value = self.value_stream(x)
            advantage = self.advantage_stream(x)
            return value + (advantage - advantage.mean(dim = -1, keepdim = True))
        else:
            return self.readout(x)      

    def predict(self, x):
        with torch.no_grad():
            return self.forward(torch.from_numpy(x.astype(np.float32))).numpy()

    def reset_noise(self):
        if self.use_noisy:
            for layer in self.layers:
                if isinstance(layer, NoisyLinear):
                    layer.reset_noise()
            if hasattr(self, "readout") and isinstance(self.readout, NoisyLinear):
                self.readout.reset_noise()