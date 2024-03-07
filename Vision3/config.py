params = dict()

params['num_classes'] = 101

params['dataset'] = './dataset'

params['epoch_num'] = 40
params['batch_size'] = 16
params['step'] = 10
params['num_workers'] = 2  # colab只提供两个
params['learning_rate'] = 1e-2  # 学习率
params['momentum'] = 0.9  # 动量
params['weight_decay'] = 1e-5  # L2正则化
params['display'] = 10  # 多少个batch打印输出一次
params['pretrained'] = None
params['gpu'] = [0]
params['log'] = 'log'  # 训练日志所在文件夹
params['save_path'] = './checkpoints'  # 模型训练保存位置
params['clip_len'] = 64
params['frame_sample_rate'] = 1  # 视频片段取样率
