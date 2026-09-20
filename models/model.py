import torch
import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.input = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

        self.batch1 = nn.BatchNorm2d(dim)

        self.activation1 = nn.SiLU()

        self.output = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

        self.batch2 = nn.BatchNorm2d(dim)

        self.activation2 = nn.SiLU()

    def forward(self, x):
        residual = x

        x = self.input(x)
        x = self.batch1(x)
        x = self.activation1(x)

        x = self.output(x)
        x = self.batch2(x)
        x = self.activation2(x)

        x += residual

        return x

class Model(nn.Module):
    def __init__(self, num_layers=1, dim=128, board_size=19):
        super().__init__()

        num_actions = board_size * board_size + 1

        self.input_proj = nn.Conv2d(in_channels=2, out_channels=dim, kernel_size=3, padding=1)

        self.layers = nn.ModuleList([ResNet(dim=dim) for _ in range(num_layers)])

        self.flatten = nn.Flatten()

        self.output_proj = nn.Linear(board_size * board_size * dim, 256)

        self.prob_output = nn.Linear(256, 1)
        self.prob = nn.Tanh()

        self.policy_head = nn.Linear(256, num_actions)

    def forward(self, x):
        x = self.input_proj(x)

        for layer in self.layers:
            x = layer(x)

        x = self.flatten(x)

        x = self.output_proj(x)

        value = self.prob_output(x)
        value = self.prob(value)

        policy = self.policy_head(x)

        return policy, value
