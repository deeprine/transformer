import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super(PositionalEncoding, self).__init__()

        self.d_model = d_model
        self.max_len = max_len

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)

        pos_idx = torch.arange(0, d_model, step=2, device=device)

        self.result = torch.zeros(max_len, d_model, device=device)
        self.result[:, 0::2] = torch.sin(pos/10000**(pos_idx/d_model))
        self.result[:, 1::2] = torch.cos(pos/10000**(pos_idx/d_model))

    def forward(self, x):
        batch, seq_size = x.size()
        result = self.result[:, :seq_size]

        return result


# data1 = torch.randn(4, 30)

# position = PositionalEncoding(512, 30)
# data = position(data1)
# print(data.shape)
# data = position(data1)
# data = position(data1)
# print(data.shape)