class DefaultConfig(object):
    # 训练相关变量
    batch_size = 32  # batch size
    print_batch_num = 1
    num_workers = 4  # how many workers for loading data
    is_shuffle = True
    epoch_num = 10
    lr = 0.001
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-4  # 损失函数
