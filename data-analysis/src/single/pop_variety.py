import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_pop_variety():
    pop = pd.read_csv("../../data/state_pop.csv")

    # 一共有116年的数据
    print(len(pop["year"].unique()))

    # 发现就是两个州 1900-1949年例没有人口数据
    pop["year"][pop["pop"].isna()].unique()

    # 准备计算平均人口
    p1 = pop.dropna(axis=0).groupby("year").aggregate({
        "state": len,
        "pop": np.sum
    })
    # 平均人口
    p_avg = p1["pop"] / p1["state"]
    plt.figure(num=1, figsize=(20, 8), dpi=80)
    plt.grid(alpha=0.5)
    plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel("年份")
    plt.ylabel("平均人口")
    plt.plot(p_avg.index, p_avg)
    plt.xticks(range(np.min(p_avg.index), np.max(p_avg.index)+1, 5), p_avg.index[::5])
    plt.show()


if __name__ == '__main__':
    get_pop_variety()
