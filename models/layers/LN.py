import torch.nn as nn
import torch

class LayerNorm(nn.Module):
    def __init__(self, features, epsilon=1e-5, gamma=1.0, beta=0.0):
        super(LayerNorm, self).__init__()

        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.Tensor(features).fill_(gamma))
        self.beta = nn.Parameter(torch.Tensor(features).fill_(beta))

    def forward(self, x, sub_x):
        x = x + sub_x

        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)

        x = (x - mean) / torch.sqrt(var + self.epsilon)

        x = self.gamma * x + self.beta

        return x

# data = torch.randn(128, 30, 512)

# LN = LayerNorm()
# result = LN(data, data)
# result.shape
