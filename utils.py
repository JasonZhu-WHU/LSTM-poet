import torch
import numpy as np
from config import Config

cfg = Config()

word2ix = np.load(cfg.word2id_file, allow_pickle=True).item()
ix2word = np.load(cfg.id2word_file, allow_pickle=True).item()
poetry_data = np.load(cfg.poetry_file, allow_pickle=True)

def print_poetry(ids):
    poetry = [ix2word[i] for i in ids]
    return poetry

# 8290: '<EOP>', 8291: '<START>', 8292: '</s>'
def filter_poetry_ids(ids):
    filtered_ids = [id for id in ids if id != cfg.padding_id]
    return filtered_ids


