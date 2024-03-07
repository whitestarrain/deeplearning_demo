import torch
from models import Generator
if __name__ == '__main__':
    model = Generator()
    model.load_state_dict(torch.load("./netN2P"))
    pass