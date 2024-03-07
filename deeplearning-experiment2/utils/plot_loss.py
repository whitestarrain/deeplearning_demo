import matplotlib.pyplot as plt


def plot_loss(train_loss):
    # loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(train_loss)), train_loss, label='train loss')
    plt.xlabel("epoch_id")
    plt.ylabel("loss")
    plt.xticks(range(len(train_loss))[::2])
    plt.grid(alpha=0.6)
    plt.legend()
    plt.show()
