import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def plot_acc_loss(train_loss, val_acc):
    # loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(train_loss)), train_loss, label='train loss')
    plt.plot(range(len(val_acc)), val_acc, label='val acc')
    plt.xlabel("epoch_id")
    plt.ylabel("loss")
    plt.xticks(range(len(train_loss)))
    plt.grid(alpha=0.6)
    plt.legend()
    plt.show()


