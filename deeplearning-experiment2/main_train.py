import torch
import torch.optim
from torch import nn, Tensor
from torch.utils.data.dataloader import DataLoader

from data.dataset import WordListDataSet
from models.LSTMModel import LSTMModel
from data.wordvector_model import get_vocab_map, get_id2vector
from config import DefaultConfig

if __name__ == '__main__':
    device = torch.device("cuda")
    word2id, id2word = get_vocab_map(DefaultConfig.embedding_model_path)
    id2vector = get_id2vector(DefaultConfig.embedding_model_path)
    train_data = WordListDataSet(DefaultConfig.data_root, DefaultConfig.pre_num, word2id)
    train_data = DataLoader(train_data, batch_size=16, shuffle=DefaultConfig.is_shuffle)
    criteon = nn.CrossEntropyLoss().to(device)
    lstm = LSTMModel(id2vector, DefaultConfig.hidden_size).to(device)
    optimizer = torch.optim.Adam(lstm.parameters(), lr=DefaultConfig.lr)

    loss_arr = []
    data_len = len(train_data)
    lstm.train()
    for i in range(DefaultConfig.epoch_num):
        epoch_loss = torch.tensor(0).float().to(device)
        for batch_id, (data, label) in enumerate(train_data):
            data, label = data.to(device), label.to(device)
            out = lstm(data)
            loss: Tensor = criteon(out, label)
            epoch_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_arr.append(epoch_loss / data_len)
        print("epochid:{},loss:{}".format(i, epoch_loss / data_len))
        torch.save(lstm.state_dict(), DefaultConfig.lstm_model_path + "/lstm_epoch{}.pth".format(i))

