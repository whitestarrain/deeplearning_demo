import os


def get_train_txt(annotation_path: str, image_prefix: str, txt_file_path: str):
    """
    :param annotation_path: 标注数据的txt文件
    :param image_prefix: 图片所在目录
    :param txt_file_path: 生成的指定格式的txt文件，存储目录
    :return:
    """
    r = open(annotation_path, "r", encoding="utf-8")
    lines = r.readlines()
    # 将<Frame Number> <Track ID Number > <Annotation Class>
    # <Bounding box top left x coordinate> <Bounding box top left y coordinate>
    # <Bounding Box Width> <Bounding Box Height>
    # 转换为 file_path x_min,y_min,x_max,y_max,class

    out = {}  # out用来保存每张图片的标注信息
    for line in lines:
        spl = line.replace("\n", "").split("\t")
        image_name = "%.6d" % int(spl[0])
        class_id = str(int(spl[2]) - 1)
        x_min = str(round(float(spl[3])))
        y_min = str(round(float(spl[4])))
        x_max = str(round(float(spl[3]) + float(spl[5])))
        y_max = str(round(float(spl[4]) + float(spl[6])))
        if image_name not in out.keys():
            out[image_name] = list()
        out[image_name].append([x_min, y_min, x_max, y_max, class_id])
    r.close()

    o = open(txt_file_path, "w")
    out_str_list = list()
    for key in out.keys():
        line = ""
        image_pth = image_prefix + "/" + key + ".jpg"
        line += image_pth
        for annotation in out[key]:
            line += " "
            line += ",".join(annotation)
        out_str_list.append(line)
    o.write("\n".join(out_str_list))
    o.close()


if __name__ == '__main__':
    get_train_txt("../dataset/Annotation/train/Seq5-IR.txt",
                  "D:/learn/PyCharm-workplace/ComputerVision1/dataset/Image/train", "../image_annotation.txt")
