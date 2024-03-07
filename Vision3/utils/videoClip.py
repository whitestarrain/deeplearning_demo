import os
import cv2
import re

pear_data_path = "./dataset/pear/hollywood/"

train_annotation = pear_data_path + "annotations/train_clean.txt"
train_out = "./dataset/train/"
val_out = "./dataset/validation/"
val_annotation = pear_data_path + "annotations/test_clean.txt"
all_videos_path = pear_data_path + "videoclips/"


def clip_video(video_path, out_path, start_frame, end_frame):
    videoCapture = cv2.VideoCapture(video_path)
    fps = 16  # 保存视频的帧率
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # 保存视频的大小

    videoWriter = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, size)

    i = 0

    if __name__ == '__main__':
        while True:
            success, frame = videoCapture.read()
            if success:
                i += 1
                if (i >= start_frame and i <= end_frame):
                    videoWriter.write(frame)
            else:
                print(video_path, 'end')
                break


def clip_videos(annotation, file_out_path):
    anno_file = open(annotation, "r", encoding="utf-8")
    lines = anno_file.readlines()
    for line in lines:
        ret = re.match(r'\"(.*)\"\((.*)\) <(.*?)>', line)
        file_name = ret.group(1)
        frame_range = ret.group(2)
        class_name = ret.group(3)
        video_path = all_videos_path + file_name
        out_path = file_out_path + class_name + "/"
        start_frame = int(frame_range.split("-")[0])
        end_frame = int(frame_range.split("-")[1])

        if (not os.path.exists(out_path)):
            os.makedirs(out_path)

        clip_video(video_path, out_path + file_name.replace(" ", ""), start_frame, end_frame)

    anno_file.close()
    pass


if __name__ == '__main__':
    clip_videos(val_annotation, val_out)
    clip_videos(train_annotation, train_out)
