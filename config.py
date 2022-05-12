class Config:
    def __init__(self):
        self.seq_len = 48


        # 与数据向量表示的维度有关参数
        self.hidden_size = 1024
        self.embedding_size = 100

        # 与文件路径有关的参数
        self.word2id_file = 'Dataset/word2ix.npy'
        self.id2word_file = 'Dataset/ix2word.npy'
        self.poetry_file = 'Dataset/data.npy'

        # 与训练和网络有关的参数
        self.using_pretrained = False
        self.bidirectional = False
        self.lstm_layers = 3
        self.batch_size = 16
        self.num_epochs = 10
        self.learning_rate = 0.001
        self.dropout_rate = 0.3
        self.step_size = 10 #学习率更新步长(每step_size个epoch更新一次)

        # 数据集相关的常量
        self.padding_id = 8292
        self.start_id = 8291
        self.end_id = 8290
        self.vocab_size = 8293      