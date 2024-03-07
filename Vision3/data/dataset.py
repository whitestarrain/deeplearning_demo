import os
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset


class VideoDataset(Dataset):

    def __init__(self, directory, mode='train', clip_len=8, frame_sample_rate=1):
        folder = Path(directory) / mode  # 获取视频所在文件夹
        self.clip_len = clip_len

        # 数据增强
        self.short_side = [128, 160]  # resize到的范围
        self.crop_size = 112  # 最终截取视频的大小
        self.frame_sample_rate = frame_sample_rate
        self.mode = mode

        self.fnames, labels = [], []
        for label in sorted(os.listdir(folder)):
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder, label, fname))
                labels.append(label)
        # 获取index到label的map
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        # 把所有labelname转换为一个数组
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        label_file = str(len(os.listdir(folder))) + 'class_labels.txt'
        with open(label_file, 'w') as f:
            for id, label in enumerate(sorted(self.label2index)):
                f.writelines(str(id + 1) + ' ' + label + '\n')

    def __getitem__(self, index):
        # 加载视频
        buffer = self.loadvideo(self.fnames[index])

        while buffer.shape[0] < self.clip_len + 2:
            index = np.random.randint(self.__len__())

        if self.mode == 'train' or self.mode == 'training':
            buffer = self.randomflip(buffer)
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)

        return buffer, self.label_array[index]

    def to_tensor(self, buffer):
        # [D, H, W, C] ==> [C, D, H, W]
        # D = Depth , H = Height, W = Width, C = Channels
        return buffer.transpose((3, 0, 1, 2))

    def loadvideo(self, fname):
        remainder = np.random.randint(self.frame_sample_rate)
        # VideoCapture读取视频
        capture = cv2.VideoCapture(fname)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # resize到指定范围内，同时要保证视频长宽比不变
        if frame_height < frame_width:
            resize_height = np.random.randint(self.short_side[0], self.short_side[1] + 1)
            resize_width = int(float(resize_height) / frame_height * frame_width)
        else:
            resize_width = np.random.randint(self.short_side[0], self.short_side[1] + 1)
            resize_height = int(float(resize_width) / frame_width * frame_height)

        # 大于300帧，就随机截取，帧数大小和取样率有关
        start_idx = 0
        end_idx = frame_count - 1
        frame_count_sample = frame_count // self.frame_sample_rate - 1
        if frame_count > 300:
            end_idx = np.random.randint(300, frame_count)
            start_idx = end_idx - 300
            frame_count_sample = 301 // self.frame_sample_rate - 1

        buffer = np.empty((frame_count_sample, resize_height, resize_width, 3), np.dtype('float32'))

        count = 0
        retaining = True
        sample_count = 0

        # 读取每一帧存入buffer
        while (count <= end_idx and retaining):
            retaining, frame = capture.read()
            if count < start_idx:
                count += 1
                continue
            if retaining is False or count > end_idx:
                break
            if count % self.frame_sample_rate == remainder and sample_count < frame_count_sample:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # resize 每一帧到指定大小
                if (frame_height != resize_height) or (frame_width != resize_width):
                    frame = cv2.resize(frame, (resize_width, resize_height))
                buffer[sample_count] = frame
                sample_count = sample_count + 1
            count += 1
        capture.release()
        return buffer

    def crop(self, buffer, clip_len, crop_size):
        time_index = np.random.randint(buffer.shape[0] - clip_len)  # 时间切片
        height_index = np.random.randint(buffer.shape[1] - crop_size)  # 高度切片
        width_index = np.random.randint(buffer.shape[2] - crop_size)  # 宽度切片

        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer

    def normalize(self, buffer):
        # 标准化
        # buffer = (buffer - 128)/128.0
        for i, frame in enumerate(buffer):
            frame = (frame - np.array([[[128.0, 128.0, 128.0]]])) / 128.0
            buffer[i] = frame
        return buffer

    def randomflip(self, buffer):
        """0.5概率水平翻转"""
        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer

    def __len__(self):
        return len(self.fnames)


if __name__ == '__main__':
    datapath = '../dataset'
    train_dataloader = \
        DataLoader(VideoDataset(datapath, mode='train'), batch_size=10, shuffle=True, num_workers=0)
    for step, (buffer, label) in enumerate(train_dataloader):
        print("label: ", label)
