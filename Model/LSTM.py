from unicodedata import bidirectional
import torch
import torch.nn as nn
from config import Config
# from fasttext_embedding import word_embedding

cfg = Config()

class Model(nn.Module):
    def __init__(self, vocab_size=cfg.vocab_size, embedding_size=cfg.embedding_size, hidden_size=cfg.hidden_size):
        super(Model, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, cfg.lstm_layers, batch_first=True, dropout=0,bidirectional=False)
        self.output = nn.Sequential(
            nn.Linear(cfg.hidden_size, 2048),
            nn.Tanh(),
            nn.Dropout(p=cfg.dropout_rate),
            nn.Linear(2048, 4096),
            nn.Tanh(),
            nn.Dropout(p=cfg.dropout_rate),
            nn.Linear(4096, cfg.vocab_size)
        )

    def forward(self, input, hidden=None):
        embeddings = self.embeddings(input)
        batch_size, seq_len = input.size()
        if hidden is None:
            h0 = input.data.new(cfg.lstm_layers, batch_size, self.hidden_size).fill_(0).float()
            c0 = input.data.new(cfg.lstm_layers, batch_size, self.hidden_size).fill_(0).float()
        else:
            h0, c0 = hidden
        output, hidden = self.lstm(embeddings, (h0, c0))
        output = self.output(output)
        output = output.reshape(batch_size*seq_len, -1)
        return output, hidden