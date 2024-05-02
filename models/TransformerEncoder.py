import torch.nn as nn
import torch

from models.layers.PE import PositionalEncoding
from models.layers.MHA import ScaleDotProductAtt
from models.layers.FFN import FeedForwardNetwork
from models.layers.LN import LayerNorm

class Encoder(nn.Module):
    def __init__(self,
                 max_len,
                 d_model,
                 num_heads,
                 batch_size,
                 vocab_size,
                 device
    ):
      super(Encoder, self).__init__()
      self.embedding = nn.Embedding(vocab_size, d_model)
      self.positional_encoding = PositionalEncoding(vocab_size, d_model, device)
      self.multi_head_attention = ScaleDotProductAtt(
                                    batch_size=batch_size,
                                    sequence_length=max_len,
                                    num_heads=num_heads,
                                    emb_size=d_model
                                  )
      self.feed_forward_layer = FeedForwardNetwork(d_model)
      self.layer_norm = LayerNorm(features=d_model)
      self.dropout1 = nn.Dropout(0.1)
      self.dropout2 = nn.Dropout(0.1)
      self.dropout3 = nn.Dropout(0.1)

    def forward(self, src, src_mask, n):
      # src -> (batch_size, seq_len)
        if n != 0:
            embedding = src

        if n == 0:
            embedding = self.embedding(src)

            positional_encoding = self.positional_encoding(src)
            positional_encoding = torch.einsum('bc->cb', positional_encoding)

            embedding += positional_encoding

        embedding = self.dropout1(embedding)
        q, k, v = embedding, embedding, embedding

        result, att = self.multi_head_attention(q, k, v, is_mask=src_mask)
        result = self.dropout2(result)
        add_norm_1 = self.layer_norm(result, embedding)

        ffn_result = self.feed_forward_layer(add_norm_1)
        ffn_result = self.dropout3(ffn_result)
        add_norm_2 = self.layer_norm(ffn_result, add_norm_1)

        return add_norm_2

# max_len = 30
# d_model = 512
# num_heads = 8
# batch_size = 4

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# input_data = torch.randint(30, (4, 30))

# encoder = Encoder(max_len, d_model, num_heads, batch_size)
# encoder(input_data)