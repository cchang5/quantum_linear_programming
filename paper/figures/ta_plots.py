import matplotlib.pyplot as plt
from qlpdb.data.models import Data as data_Data
import numpy as np

def getdata():
    nndata = {
        ta: data_Data.objects.filter(
            experiment__graph__tag=f"NN(6)",
            experiment__tag="FixEmbedding_Constant_-0.1_0.12",
            experiment__settings__annealing_time=400,
            experiment__settings__num_spin_reversal_transforms=ta
        ).to_dataframe()
        for ta in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    }
    return nndata

def plot(data):
    X = list(data.keys())
    y = [
        data[key].groupby("energy").count()["id"].iloc[0]
        / data[key].count()["id"]
        for key in X
    ]
    fig = plt.figure("scaling", figsize=(7, 4))
    ax = plt.axes([0.15, 0.15, 0.8, 0.8])
    ax.errorbar(x=X, y=y, marker="o")
    ax.set_xscale("log")
    plt.draw()
    plt.show()


if __name__=="__main__":
    data = getdata()
    plot(data)