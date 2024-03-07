import torch
import torch.nn as nn
from models.blocks.BasicConv2d import BasicConv2d


class ReductinoB(nn.Module):
    def __init__(self):
        super(ReductinoB, self).__init__()
        self.branch1 = nn.MaxPool2d(3, 2, 0)
        self.branch2 = nn.Sequential(
            BasicConv2d(1024, 192, 1, 1, 0, bias=False),
            BasicConv2d(192, 192, 3, 2, 0, bias=False)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(1024, 256, 1, 1, 0, bias=False),
            BasicConv2d(256, 256, (1, 7), (1, 1), (0, 3), bias=False),
            BasicConv2d(256, 320, (7, 1), (1, 1), (3, 0), bias=False),
            BasicConv2d(320, 320, 3, 2, 0, bias=False)
        )

    def forward(self, x):
        return torch.cat((
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
        ), 1)


if __name__ == '__main__':
    print(
        ReductinoB()(torch.randn((1, 1024, 17, 17))).shape
        # [1, 1536, 8, 8]
    )
