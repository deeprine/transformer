import torch.nn as nn
import torch

class ScaleDotProductAtt(nn.Module):
    def __init__(self,
                 batch_size=4,
                 sequence_length=50,
                 num_heads=8,
                 emb_size=512
                 ):
        super(ScaleDotProductAtt, self).__init__()

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_heads = num_heads
        self.emb_size = emb_size

        self.eq = nn.Linear(self.emb_size, self.emb_size)
        self.ek = nn.Linear(self.emb_size, self.emb_size)
        self.ev = nn.Linear(self.emb_size, self.emb_size)

        self.softmax = nn.Softmax(dim=-1)
        self.head_emb_size = self.emb_size // self.num_heads

    def forward(self, q, k, v, is_mask=None):
        q_split = q.view(self.batch_size, self.sequence_length, self.num_heads, self.head_emb_size).permute(0, 2, 1, 3)
        k_split = k.view(self.batch_size, self.sequence_length, self.num_heads, self.head_emb_size).permute(0, 2, 1, 3)
        v_split = v.view(self.batch_size, self.sequence_length, self.num_heads, self.head_emb_size).permute(0, 2, 1, 3)

        qk_t = torch.einsum('bhqd,bhkd->bhqk', q_split, k_split)
        scaled_qk_t = qk_t / torch.sqrt(torch.tensor(self.emb_size, dtype=torch.float32))

        if is_mask is not None:
            # scaled_qk_t = torch.tril(scaled_qk_t)
            scaled_qk_t.masked_fill_(is_mask == 0, -10000)

        softmax_qk_t = self.softmax(scaled_qk_t)
        result = torch.einsum('bhij,bhkl->bhjl', softmax_qk_t, v_split)

        result = result.transpose(1, 2).contiguous().view(self.batch_size, self.sequence_length,
                                                          self.num_heads * (self.emb_size // self.num_heads))

        return result, softmax_qk_t

# q = torch.randn(4, 50, 512)
# k = torch.randn(4, 50, 512)
# v = torch.randn(4, 50, 512)

# SDA = ScaleDotProductAtt()
# result, att = SDA(q, k, v, is_mask=None)
# print(result.shape)