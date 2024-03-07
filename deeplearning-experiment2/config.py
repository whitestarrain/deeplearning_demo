class DefaultConfig(object):
    # 文件路径
    embedding_model_path = "./checkpoints/word2vec_model.bin"  # 本机训练测试时，word embedding模型路径
    lstm_model_path = "./checkpoints"  # 本机训练测试时，lstm模型存放路径
    data_root = "./dataset/out/yuliao.txt"  # 本机训练测试时,语料存放路径。

    # 训练相关变量
    batch_size = 32  # batch size
    num_workers = 4  # how many workers for loading data
    is_shuffle = False
    epoch_num = 100
    lr = 0.003
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-4  # 损失函数
    pre_num = 4  # 使用多少个序列算下一个
    hidden_size = 128
