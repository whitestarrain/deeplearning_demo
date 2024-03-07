from gensim.models import word2vec


def train_save_model(yuliao_path, model_path):
    print('加载语料文件...')
    sentences = word2vec.PathLineSentences("../dataset/out/yuliao.txt")
    print(sentences)
    print('模型训练中...')
    model = word2vec.Word2Vec(sentences, min_count=1, window=3, size=20)  # 训练skip-gram模型
    # 保存模型，以便重用
    print('保存模型文件中...')
    model.save(model_path)


def load_return_model(model_path):
    return word2vec.Word2Vec.load(model_path)


def get_vocab_map(model_path):
    model = load_return_model(model_path)
    vocab = model.wv.vocab
    word2id = {}
    id2word = {}
    for v in vocab:
        word2id[v] = vocab[v].index
        id2word[vocab[v].index] = v
    return word2id, id2word


def get_id2vector(model_path):
    model = load_return_model(model_path)
    vocab = model.wv.vocab
    id2wordvector = {}
    for v in vocab:
        id2wordvector[vocab[v].index] = model.wv[v]
    return id2wordvector


# def testf(model_path):
#     model = word2vec.Word2Vec.load(model_path)
#     print(model[10])
#     items = model.wv.similar_by_vector("青年", topn=5)
#     for item in items:
#         print(item)
#     v = model.wv.vocab
#
#     print(len(v))
#     print("青年 所对应的词向量:", model.wv["青年"])
#     print("青年 所对应的字典index:", v["青年"], v["青年"].index)
#     Vocab(count:14, index:809, sample_int:4294967296)

if __name__ == '__main__':
    train_save_model("../dataset/out/yuliao.txt", "../checkpoints/word2vec_model.bin")
    # testf("../checkpoints/word2vec_model.bin")
    # get_vocab_map("../checkpoints/word2vec_model.bin")

    # model = load_return_model("../checkpoints/word2vec_model.bin")
    # print(model.wv["青年"])
    #
    # id2wordvector = get_id2vector("../checkpoints/word2vec_model.bin")
    # print(id2wordvector[809])
