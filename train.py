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
import argparse

from tqdm import tqdm

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train(arg):

    src_max_len = arg.src_max_len
    src_vocab_size = arg.src_vocab_size
    target_max_len = arg.target_max_len
    target_vocab_size = arg.target_vocab_size

    d_model = arg.d_model
    num_heads = arg.num_heads
    batch_size = arg.batch_size
    repeat_N = arg.repeat_n

    learning_rate = arg.lr
    epoch = arg.epoch

    is_train = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_model_path = arg.save_model
    dataset_train = os.path.join(arg.datasets, 'wmt14_translate_de-en_train.csv')
    dataset_valid = os.path.join(arg.datasets, 'wmt14_translate_de-en_validation.csv')
    dataset_test = os.path.join(arg.datasets, 'wmt14_translate_de-en_test.csv')

    print('train datasets path - ',dataset_train)
    print('valid datasets path - ',dataset_valid)
    print('test datasets path - ',dataset_test)

    train = pd.read_csv(dataset_train,lineterminator='\n')
    validation = pd.read_csv(dataset_valid,lineterminator='\n')
    test = pd.read_csv(dataset_test,lineterminator='\n')

    full_df = pd.concat([
        train,
        validation,
        test
    ])

    full_df = full_df[:5000]

    src_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    src_pad_token_id = src_tokenizer.pad_token_id

    target_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    target_pad_token_id = target_tokenizer.pad_token_id

    dataset = TranslationDataset(
        full_df,
        src_tokenizer,
        target_tokenizer,
        src_max_len,
        target_max_len
    )

    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Transformer(
        is_train=True,
        src_pad_token_id = src_pad_token_id,
        target_pad_token_id = target_pad_token_id,
        src_max_len=src_max_len,
        target_max_len=target_max_len,
        src_vocab_size=src_vocab_size,
        target_vocab_size=target_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        batch_size=batch_size,
        repeat_N=repeat_N,
        device=device
    )

    model.to(device)
    model.apply(initialize_weights)

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index = target_pad_token_id)
    # warmup and label smooting 구현예정

    for e in range(epoch):
        total_loss = 0
        print('epoch Number : ', e+1)
        for data in tqdm(train_dataloader):
            src_ids = data['input_ids'].to(device)
            src_attention_mask = data['input_attention_mask'].to(device)

            label_ids = data['output_ids'].to(device)
            label_attention_mask = data['output_attention_mask'].to(device)

            outputs = model(src_ids, label_ids, src_attention_mask, label_attention_mask)
            # outputs = torch.argmax(outputs, dim=-1)

            # output_reshape = outputs.contiguous().view(-1, outputs.shape[-1])
            # output_ids = output_ids[:, :].contiguous().view(-1)

            outputs = outputs.view(-1, target_vocab_size)
            label_ids = label_ids.view(-1)

            loss = criterion(outputs, label_ids)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print("Average Loss:", avg_loss)

def args():
    parser = argparse.ArgumentParser(description='image merge')

    parser.add_argument('--datasets', type=str, default='/home/yongseong/Downloads/archive/')
    parser.add_argument('--save_model', type=str, default='/home/yongseong/Downloads/archive/save_model')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--src_max_len', type=int, default=256)
    parser.add_argument('--src_vocab_size', type=int, default=200000)
    parser.add_argument('--target_max_len', type=int, default=256)
    parser.add_argument('--target_vocab_size', type=int, default=200000)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--repeat_n', type=int, default=6)
    parser.add_argument('--num_heads', type=str, default=8)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=int, default=0.001)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    arg = args()
    train(arg)

