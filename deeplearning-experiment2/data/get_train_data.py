from data.wordvector_model import *


def get_train_data(pre_num, word2id, in_path, out_path):
    """
    !! 注意，该方法只是用来测试用。具体逻辑和实现的dataset中的相同 !!
    根据分号词后的数据，获取训练数据。
    使用前几个词汇预测后面的一个词汇，
    :param pre_num:  使用多少个词汇预测下一个词汇
    :param word2id:  词汇转换为id
    :param in_path:
    :param out_path:
    :return:
    """
    with open(in_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    data = list()
    for l in lines:
        words = l.split(" ")
        words = [w for w in words if len(w) > 0]
        if len(words) < pre_num + 1:
            continue
        for i in range(pre_num, len(words) - 1):
            seq = words[i - pre_num: i + 1]
            seq = [str(word2id[w]) for w in seq]
            data.append(" ".join(seq))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(data))


if __name__ == '__main__':
    word2id, id2word = get_vocab_map("../checkpoints/word2vec_model.bin")
    get_train_data(3, word2id, "../dataset/out/yuliao.txt", "../dataset/out/train_data.txt")
