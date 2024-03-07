import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_gas_variety():
    gas = pd.read_csv("../../data/gas_price.csv")
    plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(num=1, figsize=(20, 8), dpi=80)
    plt.grid(alpha=0.5)
    plt.plot(range(len(gas)), gas["gas_current"])
    plt.xticks(range(len(gas))[::5], gas["year"][::5])
    plt.xlabel("年份")
    plt.ylabel("油价(dollar/gallon)")
    plt.show()
    pass


if __name__ == '__main__':
    get_gas_variety()
