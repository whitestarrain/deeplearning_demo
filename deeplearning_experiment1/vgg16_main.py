import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

from config import DefaultConfig
from utils.utils import train_test_split, plot_acc_loss

def train():
    device = torch.device("cuda")
    vgg16 = torchvision.models.vgg16(num_classes=2).to(device)

    # 定义transform
    img_transform = T.Compose([
        T.Resize([299, 299]),
        T.ToTensor(),
        T.Normalize([0.596, 0.595, 0.595], [0.298, 0.298, 0.298])
    ])

    # 读取全部数据
    dataset = ImageFolder(root=DefaultConfig.data_root, transform=img_transform)

    # 划分训练集和验证集
    train_dataset, val_dataset = train_test_split(dataset)

    # 封装到DataLoader中
    train_dataset = DataLoader(train_dataset, batch_size=DefaultConfig.batch_size, shuffle=DefaultConfig.is_shuffle,
                               num_workers=DefaultConfig.num_workers)
    val_dataset = DataLoader(val_dataset, batch_size=DefaultConfig.batch_size, shuffle=DefaultConfig.is_shuffle,
                             num_workers=DefaultConfig.num_workers)

    # loss计算
    criteon = nn.CrossEntropyLoss().to(device)

    # 定义优化器
    optimizer = optim.Adam(vgg16.parameters(), lr=DefaultConfig.vgg16_lr)
    # 存储每个epoch最后一个batch的loss以及每个epoch后的准确率
    train_loss = list()
    val_loss = list()
    train_acc = list()
    val_acc = list()

    for epoch_id in range(DefaultConfig.epoch_num):
        loss = torch.tensor([0])

        # 训练
        vgg16.train()

        for batch_id, (data, label) in enumerate(train_dataset):
            data, label = data.to(device), label.to(device)
            logist = vgg16(data)
            loss: torch.Tensor = criteon(logist, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print("epochid:{},batchid:{},loss:{}".format(epoch_id, batch_id, loss))
        train_loss.append(loss.item())

        # 查看正确率
        vgg16.eval()

        with torch.no_grad():

            total_correct = 0
            total_num = 0
            loss = torch.tensor([0])
            for data, label in val_dataset:
                data, label = data.to(device), label.to(device)
                logist = vgg16(data)
                loss: torch.Tensor = criteon(logist, label)
                pred = logist.argmax(dim=1)

                correct = torch.eq(pred, label).float().sum().item()
                total_correct += correct
                total_num += data.size(0)

            acc = total_correct / total_num
            val_acc.append(acc)
            val_loss.append(loss.item())
            print("epoch:{},val acc:{}".format(epoch_id, acc))

            total_correct = 0
            total_num = 0
            for data, label in train_dataset:
                data, label = data.to(device), label.to(device)
                logist = vgg16(data)
                pred = logist.argmax(dim=1)

                correct = torch.eq(pred, label).float().sum().item()
                total_correct += correct
                total_num += data.size(0)

            acc = total_correct / total_num
            train_acc.append(acc)
            print("epoch:{},train acc:{}".format(epoch_id, acc))
    plot_acc_loss(train_loss, train_acc, val_loss, val_acc)


if __name__ == '__main__':
    train()
