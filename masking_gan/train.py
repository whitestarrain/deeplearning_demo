import itertools
import sys

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torchvision.utils import save_image
from torch.utils.data import SubsetRandomSampler

from models import Generator, Discriminator

batch_size = 1
lambda_cycle = 1
lambda_identity = 2
lr = 0.0001
seed = 0

print_every = 200
n_epochs = 60
input_shape = (216, 176)
odir = 'ckpt'

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

cudnn.benchmark = True

# Init dataset
transformer = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(
        lambda img: img[:, 1:-1, 1:-1]),
    torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

dataset = torchvision.datasets.ImageFolder('data/celeba/', transformer)

labels_neg = [i for i, (_, l) in enumerate(dataset.imgs) if l == 0]
labels_pos = [i for i, (_, l) in enumerate(dataset.imgs) if l == 1]

sampler_neg = torch.utils.data.sampler.SubsetRandomSampler(labels_neg)
sampler_pos = torch.utils.data.sampler.SubsetRandomSampler(labels_pos)

pos_loader = torch.utils.data.DataLoader(dataset,
                                         sampler=sampler_pos,
                                         batch_size=batch_size,
                                         )

neg_loader = torch.utils.data.DataLoader(dataset,
                                         sampler=sampler_neg,
                                         batch_size=batch_size,
                                         )

# Init models
netDP = Discriminator().cuda()
netDN = Discriminator().cuda()
netP2N = Generator().cuda()
netN2P = Generator().cuda()

criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()
criterion_gan = nn.MSELoss()

# Init tensors
real_pos = torch.zeros(batch_size, 3,
                       input_shape[0], input_shape[1]
                       ).cuda()
real_neg = torch.zeros(batch_size, 3,
                       input_shape[0], input_shape[1]
                       ).cuda()

real_lbl = torch.zeros(batch_size, 1).cuda()
real_lbl[:, 0] = 1
fake_lbl = torch.zeros(batch_size, 1).cuda()
fake_lbl[:, 0] = -1

opt_G = optim.Adam(list(netP2N.parameters()) + list(netN2P.parameters()), lr=lr, betas=(0.5, 0.999))
opt_D = optim.Adam(list(netDN.parameters()) + list(netDP.parameters()), lr=lr, betas=(0.5, 0.999))
scheduler_G = lr_scheduler.StepLR(opt_G, step_size=10, gamma=0.317)
scheduler_D = lr_scheduler.StepLR(opt_D, step_size=10, gamma=0.317)

netDN.train()
netDP.train()
netP2N.train()
netN2P.train()

print('Training...')
for epoch in range(n_epochs):
    batch = 0

    for (pos, _), (neg, _) in zip(pos_loader, neg_loader):
        netDN.zero_grad()
        netDP.zero_grad()
        netP2N.zero_grad()
        netN2P.zero_grad()

        real_pos.copy_(pos)
        real_neg.copy_(neg)

        # Train P2N Generator
        fake_neg, mask_neg = netP2N(real_pos)
        rec_pos, _ = netN2P(fake_neg)
        fake_neg_lbl = netDN(fake_neg)

        loss_P2N_cyc = criterion_cycle(rec_pos, real_pos)
        loss_P2N_gan = criterion_gan(fake_neg_lbl, real_lbl)
        loss_N2P_idnt = criterion_identity(fake_neg, real_pos)

        # Train N2P Generator
        fake_pos, mask_pos = netN2P(real_neg)
        rec_neg, _ = netP2N(fake_pos)
        fake_pos_lbl = netDP(fake_pos)

        loss_N2P_cyc = criterion_cycle(rec_neg, real_neg)
        loss_N2P_gan = criterion_gan(fake_pos_lbl, real_lbl)
        loss_P2N_idnt = criterion_identity(fake_pos, real_neg)

        loss_G = ((loss_P2N_gan + loss_N2P_gan) * 0.5 +
                  (loss_P2N_cyc + loss_N2P_cyc) * lambda_cycle +
                  (loss_P2N_idnt + loss_N2P_idnt) * lambda_identity)

        loss_G.backward()
        opt_G.step()

        # Train Discriminators
        netDN.zero_grad()
        netDP.zero_grad()
        fake_neg_score = netDN(fake_neg.detach())
        loss_D = criterion_gan(fake_neg_score, fake_lbl)
        fake_pos_score = netDP(fake_pos.detach())
        loss_D += criterion_gan(fake_pos_score, fake_lbl)

        real_neg_score = netDN.forward(real_neg)
        loss_D += criterion_gan(real_neg_score, real_lbl)
        real_pos_score = netDP.forward(real_pos)
        loss_D += criterion_gan(real_pos_score, real_lbl)

        loss_D = loss_D * 0.25

        loss_D.backward()
        opt_D.step()

        if batch % print_every == 0 and batch > 1:
            print('Epoch #%d' % (epoch + 1))
            print('Batch #%d' % batch)

            print('Loss D: %0.3f' % loss_D.data[0] + '\t' +
                  'Loss G: %0.3f' % loss_G.data[0])
            print('Loss P2N G real: %0.3f' % loss_P2N_gan.data[0] + '\t' +
                  'Loss N2P G fake: %0.3f' % loss_N2P_gan.data[0])

            print('-' * 50)
            sys.stdout.flush()

            save_image(torch.cat([
                real_neg.cpu()[0] * 0.5 + 0.5,
                mask_pos.data.cpu()[0],
                fake_pos.data.cpu()[0] * 0.5 + 0.5], 2),
                'progress_pos.png')
            save_image(torch.cat([
                real_pos.cpu()[0] * 0.5 + 0.5,
                mask_neg.data.cpu()[0],
                fake_neg.data.cpu()[0] * 0.5 + 0.5], 2),
                'progress_neg.png')

            torch.save(netN2P, odir + '/netN2P.pth')
            torch.save(netP2N, odir + '/netP2N.pth')
            torch.save(netDN, odir + '/netDN.pth')
            torch.save(netDP, odir + '/netDP.pth')
        batch += 1

    # 学习率衰减
    scheduler_G.step()
    scheduler_D.step()
