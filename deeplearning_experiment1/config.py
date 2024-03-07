class DefaultConfig(object):
    data_root = './dataset/'  # 训练集存放路径
    load_model_path = './checkpoints/model.pth'  # 模型保存路径

    batch_size = 32  # batch size
    num_workers = 4  # how many workers for loading data
    is_shuffle = True
    epoch_num = 20
    vgg16_lr = 0.0001
    inceptionv4_lr = 0.0005
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-4  # 损失函数
