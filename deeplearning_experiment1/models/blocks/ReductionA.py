import torch
import torch.nn as nn
from models.blocks.BasicConv2d import BasicConv2d


class ReductionA(nn.Module):
    def __init__(self, k=192, l=224, m=256, n=384):
        super(ReductionA, self).__init__()
        self.branch1 = nn.MaxPool2d(3, 2, 0)
        self.branch2 = BasicConv2d(384, n, 3, 2, bias=False)
        self.branch3 = nn.Sequential(
            BasicConv2d(384, k, 1, 1, 0, bias=False),
            BasicConv2d(k, l, 3, 1, 1, bias=False),
            BasicConv2d(l, m, 3, 2, 0, bias=False)
        )

    def forward(self, x):
        return torch.cat((
            self.branch1(x),
            self.branch2(x),
            self.branch3(x)
        ), 1)


if __name__ == '__main__':
    print(ReductionA(192, 224, 256, 384)(torch.randn(1, 384, 35, 35)).shape)
