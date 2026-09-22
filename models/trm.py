import torch
import torch.nn as nn

class TRM(nn.Module):
    def __init__(self, dim, num_reccurtion):
        super().__init__()

        self.n = num_reccurtion

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1),
            #nn.BatchNorm2d(dim),
            nn.GroupNorm(num_groups=19, num_channels=dim),
            nn.SiLU(),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1),
        )

    def forward(self, x, y, z):

        for _ in range(self.n):

            z = z + self.net(x + y + z)

        y = y + self.net(y + z)

        return y, z

class TinyRecurrentModel(nn.Module):
    def __init__(self, dim=64, board_size=19):
        super().__init__()

        self.dim = dim

        self.board_size = board_size

        self.num_recurrent = 5

        num_actions = board_size * board_size + 1

        self.input_proj = nn.Conv2d(in_channels=2, out_channels=dim, kernel_size=3, padding=1)

        self.trm = TRM(dim, 5)

        self.flatten = nn.Flatten()

        self.output_proj = nn.Linear(board_size * board_size * dim, 256)

        self.prob_output = nn.Linear(256, 1)
        self.prob = nn.Tanh()

        self.policy_head = nn.Linear(256, num_actions)

    def forward(self, x):

        batch_size = x.size(0)

        x_embedding = self.input_proj(x)

        y = torch.zeros_like(x_embedding, device=x.device)
        z = torch.zeros_like(x_embedding, device=x.device)

        for step in range(self.num_recurrent):
            if step < self.num_recurrent - 1:
                with torch.no_grad():
                    y, z = self.trm(x_embedding, y, z)
                y, z = y.detach(), z.detach()
            else:
                y, z = self.trm(x_embedding, y, z)

        y = self.flatten(y)
        out = self.output_proj(y)

        policy = self.policy_head(out)
        #value = self.prob(self.prob_output(out))
        value = self.prob_output(out)


        return policy, value
