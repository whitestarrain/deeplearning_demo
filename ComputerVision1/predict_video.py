import os

import cv2
from PIL import Image

from yolo import YOLO

yolo = YOLO(cuda=False)


def predict_img(img_path: str, save_path: str):
    try:
        image = Image.open(img_path)
    except:
        print(img_path + ': Open Error! Try again!')
    else:
        r_image = yolo.detect_image(image)
        r_image.save(save_path)


def predict_imgs(img_dir_path, save_dir_path: str):
    for img_name in os.listdir(img_dir_path):
        img_path = img_dir_path + "/" + img_name
        predict_img(img_path, save_dir_path + "/" + img_name)


def img_to_video(imgs_path, video_path: str):
    filelist = os.listdir(imgs_path)
    size = Image.open(imgs_path + "/" + filelist[0]).size

    fps = 10  # 视频每秒24帧

    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"DIVX"), fps, size, True)

    for item in filelist:
        # 找到路径中所有后缀名为.png的文件，可以更换为.jpg或其它
        item = imgs_path + "/" + item
        frame = cv2.imread(item)
        video.write(frame)

    video.release()


if __name__ == '__main__':
    inpath = input("Input images' dir path:")
    outpath = input("Input the dir path of output:")

    video_path = "./dataset/out/out.avi"

    predict_imgs(inpath, outpath)
    img_to_video(outpath, video_path)
