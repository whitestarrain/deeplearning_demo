from math import floor

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.dataset import random_split


def train_test_split(dataset, train_size=0.8):
    train_size = floor(train_size * len(dataset))  # 训练集分割比例
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset


def get_mean_std(dataset, mean_std_path):
    means = [0, 0, 0]
    stdevs = [0, 0, 0]
    num_imgs = len(dataset.samples)
    for image_num in range(num_imgs):
        img = dataset[image_num][0]
        for i in range(3):
            # 一个通道的均值和标准差
            means[i] += (img[i, :, :].mean())
            stdevs[i] += (img[i, :, :].std())

    means = np.asarray(means) / num_imgs
    stdevs = np.asarray(stdevs) / num_imgs

    print("normMean = {}".format(means))
    print("normstdevs = {}".format(stdevs))

    with open(mean_std_path, 'w') as f:
        f.write("normMean = {}".format(means))
        f.write("\n")
        f.write("normstdevs = {}".format(stdevs))


def plot_acc_loss(train_loss, train_acc, val_loss, val_acc):
    # loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(train_loss)), train_loss, label='train loss')
    plt.plot(range(len(val_loss)), val_loss, label='val loss')
    plt.xlabel("epoch_id")
    plt.ylabel("loss")
    plt.xticks(range(len(train_loss)))
    plt.grid(alpha=0.6)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    # accuracies
    plt.plot(range(len(train_acc)), train_acc, label='train acc')
    plt.plot(range(len(val_acc)), val_acc, label='val acc')
    plt.xlabel("epoch_id")
    plt.ylabel("acc")
    plt.xticks(range(len(train_acc)))
    plt.grid(alpha=0.6)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_acc_loss(range(30), range(30), range(30), range(30))
