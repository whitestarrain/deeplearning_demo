import re
import jieba


def delete_date(novel_path, out_path):
    """
    每篇文章最后都会有3个空行，并在文章末写上著作日期。需要删除。
    并转换为utf-8编码
    :return:
    """
    with open(novel_path, "r", encoding="GBK") as f:
        novel_data = f.read()
        novel_data = re.sub(r".*\n\n\n\n", "", novel_data, flags=re.M)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(novel_data)


def is_unicode_char(char):
    """
    判断一个字符是不是unicode字符
    :param char:
    :return:
    """

    """判断一个unicode是否是汉字"""
    if u'\u4e00' <= char <= u'\u9fa5':
        return True
    """判断一个unicode是否是数字"""
    if u'\u0030' <= char <= u'\u0039':
        return True
    """判断一个unicode是否是英文字母"""
    if (u'\u0041' <= char <= u'\u005a') or (u'\u0061' <= char <= u'\u007a'):
        return True
    if char in ('，', '。', '：', '？', '“', '”', '！', '；', '、', '《', '》', '——'):
        return True
    return False


def filter_per_line(file_path, out_path):
    """
    针对每一行进行处理
    :return:
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 滤去空字符串
    lines = [l for l in lines if len(l) > 0]

    # 不以空格和tab开头且长度短的的基本都是小说标题，直接滤去
    lines = [l for l in lines if l.startswith((" ", "\t"))]

    # 滤去每行前面的空格和tab
    pattern = re.compile(r'[ \t]+')
    lines = [pattern.sub("", l) for l in lines]

    # 省略号效果和句号相同。
    lines = [l.replace('……', '。') for l in lines]

    # 将不是unicode字符的字符剔除
    lines = ["".join([char for char in l if is_unicode_char(char)]) for l in lines]

    # 滤去长度过短的句子。
    lines = [l for l in lines if len(l) > 3]

    # print(len(lines))
    # print(lines[50:60])

    # 为了之后方便训练，不会将堆为一行
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def filter_biaodian(cnt):
    """
    滤去标点
    :param cnt:
    :return:
    """
    pat = r'[！？｡。.＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.]+|[!"#$%&\'()*+,-./:;<=>?@\[\\\]\^\_\`\{\|\}\~0-9]+'  # 去标点
    return re.sub(pat, '', cnt)


def get_yuliao(file_path, out_path, has_biaodian: bool = False):
    lcut = []
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
        if not has_biaodian:
            data = filter_biaodian(data)
        lcut = jieba.lcut(data)
        yuliao = " ".join(lcut)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(yuliao)


def pre_process(has_biaodian: bool = False):
    """
    预处理
    :return:
    """
    delete_date("../dataset/novel.txt", "../dataset/out/novel_1.txt")
    filter_per_line("../dataset/out/novel_1.txt", "../dataset/out/novel_2.txt")
    get_yuliao("../dataset/out/novel_2.txt", '../dataset/out/yuliao.txt', has_biaodian)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "true":
            pre_process(True)
        else:
            print("如果想要得到含有标点的语料请输入:true")
    else:
        pre_process()
