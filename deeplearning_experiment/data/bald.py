import os

from PIL import Image
from torch.utils.data.dataset import Dataset


class Bald(Dataset):
    def __init__(self, tf, root='/kaggle/input/bald-classification-200k-images-celeba/Dataset', mode="train"):
        """
        :param root: 数据目录
        :param mode: train,test,val三者之一。默认为train
        """
        super(Bald, self).__init__()

        self.tf = tf

        # 根据mode选择不同的数据文件夹。
        if mode == "train":
            self.root = os.path.join(root, "Train")
        elif mode == "test":
            self.root = os.path.join(root, "Test")
        elif mode == "val":
            self.root = os.path.join(root, "Validation")

        # 设置文件夹名称到label的映射
        self.classes = {}
        for name in sorted(os.listdir(self.root)):
            if not os.path.isdir(os.path.join(self.root, name)):
                continue
            self.classes[name] = len(self.classes.keys())

        # 保存所有图片的位置
        self.imgs = []
        self.labels = []
        for name in self.classes.keys():
            templabel = self.classes[name]
            for root, dirs, filenames in os.walk(os.path.join(self.root, name)):
                for filename in filenames:
                    self.imgs += [os.path.join(root, filename)]
                    self.labels += [templabel]

    def __getitem__(self, index):
        return self.tf(Image.open(self.imgs[index])), self.labels[index]

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    data = Bald(transform)
    pass
