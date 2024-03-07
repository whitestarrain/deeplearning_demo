from torch.utils.data import Dataset
import torch


class WordListDataSet(Dataset):
    def __init__(self, yuliao_path, pre_num, word2id):
        """
        根据分好词后的数据，获取训练数据。
        使用前几个词汇预测后面的一个词汇，

        :param yuliao_path: 语料文件位置
        :param pre_num: 使用多少个单词预测后一个单词
        :param word2id: word到数字的映射
        """
        self.pre_num = pre_num
        super(WordListDataSet, self).__init__()

        with open(yuliao_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        data = list()

        # 每一行可以生成不止一个训练数据
        for l in lines:
            words = l.split(" ")
            words = [w for w in words if len(w) > 0]
            if len(words) < pre_num + 1:
                continue
            for i in range(pre_num, len(words) - 1):
                seq = words[i - pre_num: i + 1]
                seq = [word2id[w] for w in seq]
                data.append(seq)
        self.data = data  # data是二维数组,每一行都代表一条训练数据

    def __getitem__(self, index):
        # data:前几个wordid
        # label:用来预测的紧接着的一个pre_num
        return torch.tensor(self.data[index][0:self.pre_num]), torch.tensor(self.data[index][self.pre_num])

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    from data.wordvector_model import get_vocab_map

    word2id, id2word = get_vocab_map("../checkpoints/word2vec_model.bin")

    w = WordListDataSet("../dataset/out/yuliao.txt", 4, word2id)
    w_iter = iter(w)
    d, l = w_iter.__next__()
    print(d, l)

    print(100 * "-")
    from torch.utils.data import DataLoader

    loader = DataLoader(w, batch_size=16)
    loader_iter = iter(loader)
    d, l = loader_iter.__next__()
    print(d.shape)
    print(l.shape)
    # torch.Size([16, 4]) batch_size,data_size
    # torch.Size([16]) batch_size
