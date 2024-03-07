'''
predict.py有几个注意点
1、无法进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
2、如果想要保存，利用r_image.save("img.jpg")即可保存。
3、如果想要原图和分割图不混合，可以把blend参数设置成False。
4、如果想根据mask获取对应的区域，可以参考detect_image中，利用预测结果绘图的部分。
seg_img = np.zeros((np.shape(pr)[0],np.shape(pr)[1],3))
for c in range(self.num_classes):
    seg_img[:, :, 0] += ((pr == c)*( self.colors[c][0] )).astype('uint8')
    seg_img[:, :, 1] += ((pr == c)*( self.colors[c][1] )).astype('uint8')
    seg_img[:, :, 2] += ((pr == c)*( self.colors[c][2] )).astype('uint8')
'''
from PIL import Image

import re

from pspnetpredict import PSPNetPredict

pspnet = PSPNetPredict(model_path="checkpoints/Epoch1-Total_Loss1.2430-Val_Loss0.6760.pth", num_classes=20, cuda=False)

if __name__ == '__main__':
    while True:
        img = input('Input image filename:')
        img_ = re.split(r"[/\\]", img)
        img_name = img_[len(img_) - 1].split(".")[0]
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = pspnet.detect_image(image)
            r_image.show()
            r_image.save("datasets/out/images/" + img_name + ".png")
