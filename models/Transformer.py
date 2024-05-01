import torch.nn as nn
import torch

from models.TransformerEncoder import Encoder
from models.TransformerDecoder import Decoder

class Transformer(nn.Module):
  def __init__(
      self,
      is_train,
      max_len,
      d_model,
      num_heads,
      repeat_N,
      batch_size,
      vocab_size,
      device
    ):

    super(Transformer, self).__init__()
    self.encoder = Encoder(max_len, d_model, num_heads, batch_size, vocab_size, device)
    self.decoder = Decoder(max_len, d_model, num_heads, batch_size, vocab_size, device)
    self.is_train = is_train
    self.repeat_N = repeat_N
    self.device = device

  def make_trg_mask(self, trg):
        trg_pad_mask = trg.unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

  def forward(self, x, target, target_mask):
        target_mask = self.make_trg_mask(target_mask)

        for _ in range(self.repeat_N):
          encoder_output = self.encoder(x)

        for _ in range(self.repeat_N):
          decoder_output = self.decoder(encoder_output, target, target_mask)

        return decoder_output

# input = torch.randint(30, (4, 30))
# target = torch.randint(30, (4, 30))