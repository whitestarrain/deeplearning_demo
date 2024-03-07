import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_park_type():
    data = pd.read_csv("../../data/All National Parks Visitation 1904-2016.csv")
    plt.figure(num=1, figsize=(20, 8))
    plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False
    park_types = data.loc[:, ["Gnis Id", "Unit Type"]].groupby("Unit Type").count()
    x = range(len(park_types))
    bar1 = plt.bar(x, park_types["Gnis Id"], width=0.6)
    plt.grid(axis="y", alpha=0.6)
    names = park_types.index.str.replace(" ", "\n")
    i = 0
    for b in bar1:
        height = b.get_height()
        plt.text(b.get_x() + b.get_width() / 2, height + 1, str(height)+"\n"+names[i], ha='center', va='bottom')
        i += 1
    plt.ylim(0, 6000)
    plt.show()


if __name__ == '__main__':
    get_park_type()
