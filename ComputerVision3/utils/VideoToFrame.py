import cv2
import os

videos_path = "./datasets/videoclips/"
out_path = "./datasets/frames/"


def write(video_path):
    vidcap = cv2.VideoCapture(video_path)

    def getFrame(sec):
        image_path = out_path + video_name + "/image" + str(count) + ".jpg"
        image_path = image_path.replace(" ", "")

        vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        hasFrames, image = vidcap.read()
        if hasFrames:
            cv2.imwrite(image_path, image)  # save frame as JPG file
        return hasFrames

    sec = 0
    frameRate = 0.5  # //it will capture image in each 0.5 second
    count = 1
    success = getFrame(sec)
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(sec)

if __name__ == '__main__':
    videos = os.listdir(videos_path)
    for i in range(videos.__len__()):
        video_name = videos[i]
        images_path = (out_path + video_name) .replace(" ", "")
        os.mkdir(images_path)
        write(videos_path+video_name)
