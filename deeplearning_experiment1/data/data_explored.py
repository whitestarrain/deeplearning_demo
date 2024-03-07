from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder

from utils.utils import get_mean_std

if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False
    dataset = ImageFolder("./dataset/", transform=transforms.Compose([transforms.ToTensor()]))
    width = list()
    height = list()
    # print(list(dataset)[0][0].shape)
    for data, label in dataset:
        width.append(data.shape[1])
        height.append(data.shape[2])

    plt.style.use("ggplot")
    plt.hist(width, bins=20)
    plt.title("图片宽度分布")
    plt.xlabel("width")
    plt.ylabel("px")
    plt.show()
    plt.title("图片高度分布")
    plt.hist(height, bins=20)
    plt.xlabel("height")
    plt.ylabel("px")
    plt.show()

    dataset = ImageFolder("./dataset/", transform=transforms.Compose([transforms.ToTensor()]))
    get_mean_std(dataset, "./mean_std.txt")
