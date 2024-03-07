import torch
import torch.nn as nn

from models.blocks.BasicConv2d import BasicConv2d


class InceptionB(nn.Module):
    def __init__(self):
        super(InceptionB, self).__init__()
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),  # 确保长宽不变
            BasicConv2d(1024, 128, 1, 1, 0, bias=False)
        )
        self.branch2 = BasicConv2d(1024, 384, 1, 1, 0, bias=False)
        self.branch3 = nn.Sequential(
            BasicConv2d(1024, 192, 1, 1, 0, bias=False),
            BasicConv2d(192, 224, (1, 7), (1, 1), (0, 3), bias=False),
            BasicConv2d(224, 256, (1, 7), (1, 1), (0, 3), bias=False)
            # 疑问：为什么是两个(1,7)。 # 而不是一个(1,7),一个(7,1)
        )
        self.branch4 = nn.Sequential(
            BasicConv2d(1024, 192, 1, 1, 0, bias=False),
            BasicConv2d(192, 192, (1, 7), (1, 1), (0, 3), bias=False),
            BasicConv2d(192, 224, (7, 1), (1, 1), (3, 0), bias=False),
            BasicConv2d(224, 224, (1, 7), (1, 1), (0, 3), bias=False),
            BasicConv2d(224, 256, (7, 1), (1, 1), (3, 0), bias=False)
        )

    def forward(self, x):
        return torch.cat((
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x),
        ), 1)


if __name__ == '__main__':
    print(
        InceptionB()(torch.randn((1, 1024, 17, 17))).shape
        # torch.Size([1, 1024, 17, 17])
    )
