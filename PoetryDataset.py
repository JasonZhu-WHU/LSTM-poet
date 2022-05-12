import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from config import Config
from utils import filter_poetry_ids
from tqdm import tqdm

cfg = Config()
word2ix = np.load(cfg.word2id_file, allow_pickle=True).item()
ix2word = np.load(cfg.id2word_file, allow_pickle=True).item()
poetry_data = np.load(cfg.poetry_file, allow_pickle=True)


class PoetryDataset(Dataset):
    def __init__(self):
        self.seq_len = cfg.seq_len
        self.config = cfg
        self.data = poetry_data
        self.ix2word = ix2word
        self.word2ix = word2ix
        self.filtered_data = self.filter_data(poetry_data)
    
    def __getitem__(self,id):
        text = self.filtered_data[id*cfg.seq_len:(id+1)*cfg.seq_len]
        labels = self.filtered_data[id*cfg.seq_len+1:(id+1)*cfg.seq_len+1]
        text = torch.from_numpy(np.array(text)).long()
        labels = torch.from_numpy(np.array(labels)).long()
        return text, labels
    
    def __len__(self):
        return int(len(self.filtered_data) / cfg.seq_len)
    
    def filter_data(self, poetry_data):
        filtered_ids = [filter_poetry_ids(ids) for ids in self.data]
        res = []
        for ids in filtered_ids:
            res.extend(ids)
        return res


if __name__ == '__main__':
    data = PoetryDataset()
    print(len(data))
    print(data[0])
