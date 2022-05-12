import torch
import torch.nn as nn
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

# 参考加载glove写法：https://blog.csdn.net/lrt366/article/details/90405146
def CreateEmbedding(using_pretrained):
    if not using_pretrained: return nn.Embedding(cfg.vocab_size, cfg.embedding_size)
    
    import fasttext
    fasttext_model = fasttext.train_unsupervised('Dataset/poem.txt')
    embedding = nn.Embedding(cfg.vocab_size, cfg.embedding_size)
    weight = torch.zeros(cfg.vocab_size, cfg.embedding_size)
    for i in range(cfg.vocab_size):
        word = ix2word[i]
        if word in fasttext_model:
            weight[i, :] = torch.from_numpy(fasttext_model.get_word_vector(word))
    embedding = nn.Embedding.from_pretrained(weight)
    embedding.weight.requires_grad = False
    return embedding