import math
import torch
import torch.nn as nn

class NoisyLinear(nn.Module):

    def __init__(self, in_features, out_features, sigma_init = 0.017):      # Default to 0.017 from NoisyNet paper
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = sigma_init

        # Mean (µ)
        self.mu_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.mu_bias = nn.Parameter(torch.Tensor(out_features))

        # Noise Std (σ)
        self.sigma_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigma_bias = nn.Parameter(torch.Tensor(out_features))

        # Noise buffers (Not trainable, but used in forward pass)
        self.register_buffer("eps_weight", torch.zeros(out_features, in_features))
        self.register_buffer("eps_bias", torch.zeros(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        
        torch.nn.init.kaiming_uniform_(self.mu_weight, nonlinearity='relu')     # He initialization
        torch.nn.init.zeros_(self.mu_bias)      # Bias 0 initially
        
        self.sigma_weight.data.fill_(self.sigma)
        self.sigma_bias.data.fill_(self.sigma)


    """def reset_parameters(self):

        bound = 1 / math.sqrt(self.in_features)
        self.mu_weight.data.uniform_(-bound, bound)
        self.mu_bias.data.uniform_(-bound, bound)
        self.sigma_weight.data.fill_(self.sigma)
        self.sigma_bias.data.fill_(self.sigma)"""

    def reset_noise(self):

        self.eps_weight = torch.randn(self.out_features, self.in_features).to(self.mu_weight.device)
        self.eps_bias = torch.randn(self.out_features).to(self.mu_bias.device)

    def forward(self, x):
        if self.training:       # Only apply noise during training
            weight = self.mu_weight + self.sigma_weight * self.eps_weight
            bias = self.mu_bias + self.sigma_bias * self.eps_bias
        else:
            weight = self.mu_weight
            bias = self.mu_bias
        
        return torch.nn.functional.linear(x, weight, bias)
