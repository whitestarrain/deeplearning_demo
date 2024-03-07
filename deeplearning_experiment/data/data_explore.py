from torchvision import transforms
from torchvision.datasets import ImageFolder
from matplotlib.pyplot import plot as plt

dataset = ImageFolder("/kaggle/input/bald-classification-200k-images-celeba/Dataset/Test",
                      transform=transforms.Compose([transforms.ToTensor()]))
width = list()
height = list()
# print(list(dataset)[0][0].shape)
for data, label in dataset:
    width.append(data.shape[1])
    height.append(data.shape[2])

plt.style.use("ggplot")
plt.hist(width, bins=20)
plt.title("width")
plt.xlabel("width")
plt.ylabel("px")
plt.show()
plt.title("height")
plt.hist(height, bins=20)
plt.xlabel("height")
plt.ylabel("px")
plt.show()
