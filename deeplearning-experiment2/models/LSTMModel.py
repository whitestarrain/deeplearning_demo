import torch.nn as nn
from torch import Tensor
import torch


class LSTMModel(nn.Module):
    def __init__(self, id2vector, hidden_dim):
        """
        :param id2vector: 事先训练好的词向量,字典格式。藉此可以得到字典长度以及embedding_dim
        :param hidden_dim: hidden层维度
        """
        super(LSTMModel, self).__init__()
        # 将id2vector 转换为tensor，用来初始化embedding
        data = []
        for i in range(len(id2vector)):
            data.append(id2vector[i])
        data = torch.tensor(data)
        # print(data.shape)

        self.embedding = nn.Embedding(len(id2vector), len(data[0]))
        self.embedding.weight.data.copy_(data)
        self.lstm = nn.LSTM(self.embedding.embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, len(id2vector))
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: Tensor):
        """
        :param x: 传入的x应该是数值类型的list，在这里会转换成词向量
        :return:
        """
        # x:[batch_size,seq] -> [seq,batch_size,embbding_size]

        # 单词编码为embedding
        # embedding需要输入long类型的变量
        x = x.long()
        # [batch_size,seq]=>[batch_size,seq,embedding_size]
        x = self.dropout(self.embedding(x))

        # output: [b,seq, hid_dim]
        # hidden/h: [num_layers,b, hid_dim]
        # cell/c: [num_layers*2,b, hid_dim]
        output, (hidden, cell) = self.lstm(x)
        hidden = hidden[self.lstm.num_layers - 1]  # 取最后一层的

        # out为最后时间点的hidden层，设计整个时间序列的信息。因此用来生成接下来一个时间步的wordid
        # [batch_size,layers,hid_dim] => [batch_size,vocab_size]
        next_wordids = self.fc(hidden)
        return next_wordids


if __name__ == '__main__':
    from data.wordvector_model import get_id2vector

    id2vector = get_id2vector("../checkpoints/word2vec_model.bin")

    """
    # 测试embedding层使用
    data = []
    for i in range(len(id2vector)):
        data.append(id2vector[i])
    data = torch.tensor(data)
    # print(data.shape)

    embedding = nn.Embedding(len(id2vector), len(data[0]))
    embedding.weight.data.copy_(data)
    # [seq,batch,1]
    import numpy as np

    x = torch.LongTensor(np.arange(64).reshape((4, 16)))

    print(embedding(x).shape)
    """
    import numpy as np

    rnn = LSTMModel(id2vector, 64)
    x = torch.LongTensor(np.arange(64).reshape((16, 4)))
    output = rnn(x)
    print(output.shape)
