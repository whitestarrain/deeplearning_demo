import torch
import torch.nn as nn

from models.blocks.BasicConv2d import BasicConv2d


class Stem(nn.Module):
    def __init__(self):
        super(Stem, self).__init__()
        self.layer1 = nn.Sequential(
            BasicConv2d(3, 32, 3, 2, 0, bias=False),
            BasicConv2d(32, 32, 3, 1, 0, bias=False),
            BasicConv2d(32, 64, 3, 1, 1, bias=False)  # same卷积
        )
        self.layer2_pool = nn.MaxPool2d(3, 2, 0)
        self.layer2_conv = BasicConv2d(64, 96, 3, 2, 0, bias=False)
        self.layer3_left = nn.Sequential(
            BasicConv2d(160, 64, 1, 1, 0, bias=False),
            BasicConv2d(64, 96, 3, 1, 0, bias=False)
        )
        self.layer3_right = nn.Sequential(
            BasicConv2d(160, 64, 1, 1, 0, bias=False),
            BasicConv2d(64, 64, (7, 1), (1, 1), (3, 0), bias=False),  # same卷积，所以padding为3
            BasicConv2d(64, 64, (1, 7), (1, 1), (0, 3), bias=False),  # same卷积，所以padding为3
            BasicConv2d(64, 96, 3, 1, 0, bias=False)
        )
        self.layer4_conv = BasicConv2d(192, 192, 3, 2, 0, bias=False)
        self.layer4_pool = nn.MaxPool2d(3, 2, 0)

    def forward(self, x):
        layer1_out = self.layer1(x)
        layer2_out_conv = self.layer2_conv(layer1_out)
        layer2_out_pool = self.layer2_pool(layer1_out)
        layer2_out = torch.cat((layer2_out_conv, layer2_out_pool), 1)
        layer3_out_left = self.layer3_left(layer2_out)
        layer3_out_right = self.layer3_right(layer2_out)
        layer3_out = torch.cat((layer3_out_left, layer3_out_right), 1)
        out = torch.cat((self.layer4_conv(layer3_out), self.layer4_pool(layer3_out)), 1)
        return out


if __name__ == '__main__':
    print(Stem()(torch.randn((1, 3, 299, 299))).shape)
