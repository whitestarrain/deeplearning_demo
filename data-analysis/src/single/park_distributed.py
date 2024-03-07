import os

os.environ["PROJ_LIB"] = "D:\learn\anaconda3\share"
import numpy as np
import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt


def point_park(a, m):
    xpt, ypt = m(a[0], a[1])
    m.plot(xpt, ypt, 'b^', markersize=5)


def get_park_distributed():
    locations = pd.read_csv("../../data/locations.csv")
    plt.figure(figsize=(20, 8))
    m = Basemap(projection='mill',
                llcrnrlat=15,
                llcrnrlon=-180,
                urcrnrlat=75,
                urcrnrlon=-20,
                resolution='l')
    m.drawcoastlines()
    m.drawcountries(linewidth=2)
    m.drawcounties(color='darkred')
    # m.etopo()
    locations.loc[:, ["lon", "lat"]].apply(point_park, axis=1, args=(m,))
    plt.show()


if __name__ == '__main__':
    get_park_distributed()
