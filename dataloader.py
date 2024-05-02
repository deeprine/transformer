import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# 데이터셋 클래스 정의
class TranslationDataset(Dataset):
    def __init__(self,
             dataframe,
             src_tokenizer,
             target_tokenizer,
             src_max_length,
             target_max_length,
        ):

        self.data = dataframe
        self.src_tokenizer = src_tokenizer
        self.target_tokenizer = target_tokenizer
        self.src_max_length = src_max_length
        self.target_max_length = target_max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text_en = self.data['en'][idx]
        text_dn = self.data['de'][idx]

        # 토큰화 및 패딩
        inputs = self.src_tokenizer(text_en,
                                padding='max_length',
                                truncation=True,
                                max_length=self.src_max_length,
                                return_tensors='pt')

        outputs = self.target_tokenizer(text_dn,
                                 padding='max_length',
                                 truncation=True,
                                 max_length=self.target_max_length,
                                 return_tensors='pt')

        input_ids = inputs['input_ids'].squeeze()
        input_attention_mask = inputs['attention_mask'].squeeze()

        output_ids = outputs['input_ids'].squeeze()
        output_attention_mask = outputs['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'input_attention_mask': input_attention_mask,
            'output_ids': output_ids,
            'output_attention_mask': output_attention_mask
        }