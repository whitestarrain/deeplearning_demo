import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable

from torch.utils import data


class hollywoodDataset(data.Dataset):
    def __init__(self):
        super(hollywoodDataset, self).__init__()
        self.labels = self.label_init()

    def __getitem__(self, item):
        pass

    def label_init(self):

        return ""
