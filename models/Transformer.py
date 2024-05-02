import torch.nn as nn
import torch

from models.TransformerEncoder import Encoder
from models.TransformerDecoder import Decoder

class Transformer(nn.Module):
  def __init__(
      self,
      is_train,
      src_pad_token_id,
      target_pad_token_id,
      src_max_len,
      target_max_len,
      src_vocab_size,
      target_vocab_size,
      d_model,
      num_heads,
      repeat_N,
      batch_size,
      device
    ):

    super(Transformer, self).__init__()
    self.encoder = Encoder(src_max_len, d_model, num_heads, batch_size, src_vocab_size, device)
    self.decoder = Decoder(target_max_len, d_model, num_heads, batch_size, target_vocab_size, device)
    self.is_train = is_train
    self.repeat_N = repeat_N
    self.device = device

    self.src_pad_token_id = src_pad_token_id
    self.target_pad_token_id = target_pad_token_id

  def make_src_mask(self, src):
        src_pad_mask = (src != self.src_pad_token_id).unsqueeze(1).unsqueeze(3)
        return src_pad_mask

  def make_trg_mask(self, trg):
        trg_pad_mask = trg.unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

  def forward(self, src, target, src_mask, target_mask):

        src_pad_mask = self.make_src_mask(src_mask)
        target_pad_mask = self.make_src_mask(target_mask)
        target_mask = self.make_trg_mask(target_mask)

        for n in range(self.repeat_N):
            src = self.encoder(src, src_pad_mask, n)

        for n in range(self.repeat_N):
            target = self.decoder(src, target, target_mask, target_pad_mask, n, self.repeat_N)

        return target

# input = torch.randint(30, (4, 30))
# target = torch.randint(30, (4, 30))