import jieba
import torch

from config import DefaultConfig
from data.wordvector_model import get_vocab_map, get_id2vector
from models.LSTMModel import LSTMModel


def word2idfun(word2id, word):
    if word in word2id:
        return word2id[word]
    else:
        return word2id["，"]


if __name__ == '__main__':
    input_words = "你看也没有学问也不懂道理"
    generate_len = 100
    word2vec_path = "./checkpoints/colab上跑出来的模型/lstm_model_without_biaodian/word2vec_model.bin"
    lstm_model_path = "./checkpoints/colab上跑出来的模型/lstm_model_without_biaodian/lstm_epoch99.pth"

    word2id, id2word = get_vocab_map(word2vec_path)
    id2vector = get_id2vector(word2vec_path)
    lstm = LSTMModel(id2vector, DefaultConfig.hidden_size)
    lstm.load_state_dict(torch.load(lstm_model_path, map_location=lambda storage, loc: storage))  # 将GPU上的模型load到cpu上

    lstm.eval()
    generate = list()
    input_words = jieba.lcut(input_words)
    input_words = [word2idfun(word2id, w) for w in input_words]
    for i in input_words:
        generate.append(i)
    input_words = input_words[len(input_words) - DefaultConfig.pre_num:]  # 根据训练模型截取
    input_words = torch.tensor(input_words)
    for i in range(generate_len):
        out = lstm(input_words.unsqueeze(0))
        next = out.argmax()
        generate.append(int(next))
        input_words = torch.cat([input_words[1:], torch.tensor([next])])
    print("".join([id2word[id] for id in generate]))
