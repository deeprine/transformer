import torch.nn as nn
import torch.optim as optim
import torch
torch.set_printoptions
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer
from dataloader import TranslationDataset
from models.Transformer import Transformer
from utils import *

import pandas as pd
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

max_len = 256
d_model = 512
num_heads = 8
batch_size = 4
repeat_N = 6
vocab_size = 200000
is_train = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

full_df = pd.concat([
    pd.read_csv('/home/yongseong/Downloads/archive/wmt14_translate_de-en_train.csv',lineterminator='\n'),
    pd.read_csv('/home/yongseong/Downloads/archive/wmt14_translate_de-en_validation.csv',lineterminator='\n'),
    pd.read_csv('/home/yongseong/Downloads/archive/wmt14_translate_de-en_test.csv',lineterminator='\n')
])

full_df = full_df[:5000]

# DataLoader에 사용할 토크나이저 초기화
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

dataset = TranslationDataset(full_df, tokenizer, max_len)
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# DataLoader를 통한 데이터 샘플 확인
# for batch in train_dataloader:
#     input_ids = batch['input_ids']
#     input_attention_mask = batch['input_attention_mask']
#
#     output_ids = batch['output_ids']
#     output_attention_mask = batch['output_attention_mask']

    # print(input_ids)
    # print("Input IDs Shape:", input_ids.shape)
    # print("Attention Mask Shape:", input_attention_mask.shape)
    # break  # 첫 번째 배치만 확인

model = Transformer(
    is_train=True,
    max_len=max_len,
    d_model=d_model,
    num_heads=num_heads,
    batch_size=batch_size,
    repeat_N=repeat_N,
    vocab_size=vocab_size,
    device=device
)

model.to(device)
model.apply(initialize_weights)

model.train()
# Adam 옵티마이저 초기화
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

epoch = 10

# warmup and label smooting 구현

for e in range(epoch):
    total_loss = 0

    for data in train_dataloader:
        src_ids = data['input_ids'].to(device)
        src_attention_mask = data['input_attention_mask'].to(device)

        label_ids = data['output_ids'].to(device)
        label_attention_mask = data['output_attention_mask'].to(device)

        # print(input_attention_mask)
        # print(output_attention_mask)

        outputs = model(src_ids, label_ids, label_attention_mask)
        # outputs = torch.argmax(outputs, dim=-1)

        # output_reshape = outputs.contiguous().view(-1, outputs.shape[-1])
        # output_ids = output_ids[:, :].contiguous().view(-1)

        outputs = outputs.view(-1, vocab_size)
        label_ids = label_ids.view(-1)

        loss = criterion(outputs, label_ids)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    print("Average Loss:", avg_loss)