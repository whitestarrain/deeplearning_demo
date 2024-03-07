import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch.utils.data import WeightedRandomSampler, DataLoader
from torchvision import models
from torchvision.transforms import transforms
from models.Flatten import Flatten

from config import DefaultConfig
from data.bald import Bald


def train(model, optimizer, criterion, train_data_loader, val_data_loader):
    loss_list = []
    val_acc = []
    for epochid in range(DefaultConfig.epoch_num):
        for batchid, (data, label) in enumerate(train_data_loader):
            data, label = data.cuda(), label.cuda()
            output = model(data)
            loss = criterion(output, label).cuda()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("#Epoch:{},Batch:{}".format(epochid, batchid))
            print("loss:{}".format(loss.item()))
            loss_list += [loss.item()]

            with torch.no_grad():
                pre_all_val = []
                y_all_val = []
                for x, y in val_data_loader:
                    x = x.cuda()
                    out: torch.Tensor = model(x)
                    pre_all_val += out.argmax(dim=1).cpu().tolist()
                    y_all_val += y.tolist()
                acc = metrics.accuracy_score(y_all_val, pre_all_val)
                val_acc += [acc]
                print("acc:{}".format(acc))
                print("=" * 30)


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # train数据集过大。这里不使用
    train_data = Bald(transform, mode="test")
    val_data = Bald(transform, mode="val")

    # 获取train权重
    train_target = np.array(train_data.labels)
    class_sample_count = np.array(
        [(train_target == t).sum() for t in sorted(np.unique(train_target))])

    weight = 1. / class_sample_count
    samples_weight_train = [weight[t] for t in train_target]

    # 获取val权重
    val_target = np.array(val_data.labels)
    class_sample_count = np.array(
        [(val_target == t).sum() for t in sorted(np.unique(val_target))])

    weight = 1. / class_sample_count
    samples_weight_val = [weight[t] for t in train_target]

    # 创建取样器，解决样本不均衡问题
    train_sampler = WeightedRandomSampler(samples_weight_train, len(samples_weight_train))

    # 验证集太大，1不使用全部数据进行验证。
    val_sampler = WeightedRandomSampler(samples_weight_val, len([i for i in val_target if i == 0]) * 2)

    train_data_loader = DataLoader(train_data, batch_size=DefaultConfig.batch_size, sampler=train_sampler,
                                   num_workers=0)
    val_data_loader = DataLoader(val_data, batch_size=DefaultConfig.batch_size, sampler=val_sampler, num_workers=0)

    # 使用预训练模型
    resnet = models.resnet18(pretrained=True)
    resnet = nn.Sequential(*list(resnet.children())[:-1],
                           Flatten(),
                           nn.Linear(512, 2)
                           ).cuda()

    optimizer = torch.optim.Adam(resnet.parameters(), lr=DefaultConfig.lr, weight_decay=DefaultConfig.weight_decay)
    criterion = torch.nn.CrossEntropyLoss().cuda()

    train(resnet, optimizer, criterion, train_data_loader, val_data_loader)
    models.resnet18()
