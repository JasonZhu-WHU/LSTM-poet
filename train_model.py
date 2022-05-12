from sched import scheduler
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from config import Config
from PoetryDataset import word2ix, ix2word, poetry_data, PoetryDataset
from Model.LSTM import Model as LSTM
from Model.BiLSTM import Model as BiLSTM
from Model.pretrained_LSTM import Model as pretrained_LSTM

cfg = Config()
poetry_dataset = PoetryDataset()
poetry_loader = DataLoader(poetry_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

def train():
    model = BiLSTM() if cfg.bidirectional else LSTM()
    if cfg.using_pretrained: model = pretrained_LSTM()
    model.train()
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(cfg.num_epochs):
        print("Epoch:"  + str(epoch))
        train_loss = 0
        for i, data in enumerate(tqdm(poetry_loader), 0):
            texts, labels = data[0].cuda(), data[1].cuda()
            labels = labels.view(-1)
            optimizer.zero_grad()
            outputs, hidden = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if (i+1) % 500 == 0:
                print('\t loss:{:.4f}'.format(loss.item()))
        scheduler.step()
    
    torch.save(model.state_dict(), 'LSTM.pth')


if __name__ == '__main__':
    train()
    



