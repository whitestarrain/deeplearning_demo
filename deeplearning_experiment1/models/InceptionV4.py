import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from models.blocks import *


class InceptionV4(nn.Module):
    def __init__(self, num_classes):
        super(InceptionV4, self).__init__()
        self.num_classes = num_classes
        self.stem = Stem()
        self.inceptionA = nn.Sequential(*self.get_4_inception_a())
        self.reductinA = ReductionA()
        self.inceptionB = nn.Sequential(*self.get_7_inception_b())
        self.reductinB = ReductinoB()
        self.inceptionC = nn.Sequential(*self.get_3_inception_c())
        self.fc = nn.Linear(1536, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.inceptionA(x)
        x = self.reductinA(x)
        x = self.inceptionB(x)
        x = self.reductinB(x)
        x = self.inceptionC(x)
        x = F.avg_pool2d(x, (x.shape[2], x.shape[3]))
        x = F.dropout(x, 0.2, training=self.training)
        x = self.fc(x.view(x.size(0), -1))
        return x
        # return F.softmax(out, dim=1)  # 交叉熵运算中包含softmax，所以这里不再根据模型进行sftmax运算

    @staticmethod
    def get_4_inception_a():
        layers = list()
        for i in range(4):
            layers.append(InceptionA())
            print(i)
        return layers

    @staticmethod
    def get_7_inception_b():
        layers = list()
        for _ in range(7):
            layers.append(InceptionB())
        return layers

    @staticmethod
    def get_3_inception_c():
        layers = list()
        for _ in range(3):
            layers.append(InceptionC())
        return layers


if __name__ == '__main__':
    summary(InceptionV4(2), (3, 299, 299))
    model = InceptionV4(2)
    print(model(torch.randn(4,3,299,299)))
