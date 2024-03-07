import torch
import torch.nn as nn
from models.blocks.BasicConv2d import BasicConv2d


class InceptionA(nn.Module):
    def __init__(self):
        super(InceptionA, self).__init__()
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),  # 加上padding确保长宽不变
            BasicConv2d(384, 96, 1, 1, 0, bias=False),
        )
        self.branch2 = BasicConv2d(384, 96, 1, 1, 0, bias=False)
        self.branch3 = nn.Sequential(
            BasicConv2d(384, 64, 1, 1, 0, bias=False),
            BasicConv2d(64, 96, 3, 1, 1, bias=False)  # 加上padding确保长宽不变
        )
        self.branch4 = nn.Sequential(
            BasicConv2d(384, 64, 1, 1, 0, bias=False),
            BasicConv2d(64, 96, 3, 1, 1, bias=False),  # 加上padding确保长宽不变
            BasicConv2d(96, 96, 3, 1, 1, bias=False)  # 加上padding确保长宽不变
        )

    def forward(self, x):
        return torch.cat((self.branch1(x),
                          self.branch2(x),
                          self.branch3(x),
                          self.branch4(x)), 1)


if __name__ == '__main__':
    print(InceptionA()(torch.randn(1, 384, 35, 35)).shape)
