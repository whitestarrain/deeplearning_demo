import torch
import torch.nn as nn
from models.blocks.BasicConv2d import BasicConv2d


class InceptionC(nn.Module):
    def __init__(self):
        super(InceptionC, self).__init__()
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),  # 添加padding保证长宽不变
            BasicConv2d(1536, 256, 1, 1, 0, bias=False)
        )
        self.branch2 = BasicConv2d(1536, 256, 1, 1, 0, bias=False)
        self.branch3_1 = BasicConv2d(1536, 384, 1, 1, 0, bias=False)
        self.branch3_2_left = BasicConv2d(384, 256, (1, 3), (1, 1), (0, 1), bias=False)
        self.branch3_2_right = BasicConv2d(384, 256, (3, 1), (1, 1), (1, 0), bias=False)
        self.branch4_1 = nn.Sequential(
            BasicConv2d(1536, 384, 1, 1, 0, bias=False),
            BasicConv2d(384, 448, (1, 3), (1, 1), (0, 1), bias=False),
            BasicConv2d(448, 512, (3, 1), (1, 1), (1, 0), bias=False),
        )
        self.branch4_2_left = BasicConv2d(512, 256, (1, 3), (1, 1), (0, 1), bias=False)
        self.branch4_2_right = BasicConv2d(512, 256, (3, 1), (1, 1), (1, 0), bias=False)

    def forward(self, x):
        temp3 = self.branch3_1(x)
        temp4 = self.branch4_1(x)
        return torch.cat(
            (
                self.branch1(x),
                self.branch2(x),
                self.branch3_2_left(temp3),
                self.branch3_2_right(temp3),
                self.branch4_2_left(temp4),
                self.branch4_2_right(temp4)
            ), 1)


if __name__ == '__main__':
    print(
        InceptionC()(torch.randn((1, 1536, 8, 8))).shape
    )
