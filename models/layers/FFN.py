import torch.nn as nn

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, df=4):
        super(FeedForwardNetwork, self).__init__()

        self.linear_1 = nn.Linear(d_model, d_model*df)
        self.linear_2 = nn.Linear(d_model*df, d_model)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)

        return x

# data = torch.randn(128, 30, 512)

# FFN = FeedForwardNetwork(512)
# result = FFN(data)
# print(result.shape)