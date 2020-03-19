import matplotlib.pyplot as plt
from qlpdb.data.models import Data as data_Data
import numpy as np
import math
import yaml

def getdata():
    nndata = {
        v: data_Data.objects.filter(
            experiment__graph__tag=f"NN(8)",
            experiment__tag=f"FixEmbedding_NegShiftLinear_-{v}_{v}",
            experiment__settings__annealing_time=400,
            experiment__settings__num_spin_reversal_transforms=10,
        ).to_dataframe()
        for v in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    }
    return nndata


def plot(data):
    X = list(data.keys())
    y = [
        data[key].groupby("energy").count()["id"].iloc[0] / data[key].count()["id"]
        for key in X
    ]
    fig = plt.figure("scaling", figsize=(7, 4))
    ax = plt.axes([0.15, 0.15, 0.8, 0.8])
    ax.errorbar(x=X, y=y, marker="o")
    # ax.set_xscale("log")
    plt.draw()
    plt.show()


def getdata_full(min, rangex):
    nndata = {
        offx: {
            int(v): data_Data.objects.filter(
                experiment__graph__tag=f"NN({v})",
                experiment__tag=f"FixEmbedding_{offx}_{min}_{rangex}",
                experiment__settings__annealing_time=400,
                experiment__settings__num_spin_reversal_transforms=10,
            ).to_dataframe()
            for v in range(2, 17)
        }
        for offx in ["NegBinary", "Constant", "Binary"]
    }
    return nndata


def degenfcn(n):
    if n % 3 == 2:
        return n // 3 + 2
    if n % 3 == 1:
        return 2 * (n // 3 + 1)
    if n % 3 == 0:
        return 1


def plot_full(data, save_tag):
    yhmc = np.array(
        [1, 0.738, 0.823, 0.657, 0.271, 0.568, 0.336, 0.1, 0.318, 0.158, 0.029, 0.169]
    )
    Xhmc = np.arange(len(yhmc)) + 2

    fig = plt.figure("scaling", figsize=(7, 4))
    ax = plt.axes([0.15, 0.15, 0.8, 0.8])
    y = dict()
    for offset in data.keys():
        X = list(data[offset].keys())
        y[offset] = []
        for key in X:
            mds = math.ceil(key / 3)
            try:
                energy_count = (
                    data[offset][key].groupby("energy").count()["id"].loc[mds]
                )
            except:
                energy_count = 0
            total_count = data[offset][key].count()["id"]
            y[offset].append(energy_count / total_count)
        ax.errorbar(x=X, y=y[offset], label=offset)
    ax.errorbar(x=Xhmc, y=yhmc, label="HMC")
    degeneracy = np.array([degenfcn(Xi) for Xi in X])
    rguess = np.array([1 / 2 ** xi for xi in X]) * degeneracy
    ax.errorbar(x=X, y=rguess, label="Random Guess")
    ax.set_yscale("log")
    ax.legend()
    plt.draw()
    plt.savefig(f"./scaling_full_{save_tag}.pdf")

    fig = plt.figure("ratio scaling", figsize=(7, 4))
    ax = plt.axes([0.15, 0.15, 0.8, 0.8])
    X = list(data["Constant"].keys())
    ax.errorbar(
        x=X,
        y=np.array(y["NegBinary"]) / np.array(y["Constant"]),
        label="NegBin / Const.",
    )
    ax.errorbar(
        x=X,
        y=np.array(y["Constant"]) / np.array(y["Constant"]),
        label="Const. / Const.",
    )
    ax.errorbar(
        x=X, y=np.array(y["Binary"]) / np.array(y["Constant"]), label="Bin / Const."
    )
    ax.axhline(y=1)
    ax.legend()
    plt.draw()
    plt.savefig(f"./scaling_full_ratio_{save_tag}.pdf")

    fig = plt.figure("ratio scaling random", figsize=(7, 4))
    ax = plt.axes([0.15, 0.15, 0.8, 0.8])
    X = list(data["Constant"].keys())
    ax.errorbar(x=X, y=np.array(y["NegBinary"]) / rguess, label="NegBin / Random")
    ax.errorbar(x=X, y=np.array(y["Constant"]) / rguess, label="Const. / Random")
    ax.errorbar(x=X, y=np.array(y["Binary"]) / rguess, label="Bin / Random")
    ax.errorbar(x=Xhmc, y=yhmc / rguess[:len(yhmc)], label="HMC / Random")
    ax.axhline(y=1)
    ax.legend()
    # ax.set_yscale("log")

    plt.draw()
    plt.savefig(f"./scaling_full_ratio_random_{save_tag}.pdf")
    plt.show()


def get_y(data):
    y = dict()
    for offset in data.keys():
        X = list(data[offset].keys())
        y[offset] = []
        for key in X:
            mds = math.ceil(key / 3)
            try:
                energy_count = (
                    data[offset][key].groupby("energy").count()["id"].loc[mds]
                )
            except:
                energy_count = 0
            total_count = data[offset][key].count()["id"]
            y[offset].append(energy_count / total_count)
    return X, y


def plot_compare(data1, data2):
    fig = plt.figure("scaling compare", figsize=(7, 4))
    ax = plt.axes([0.15, 0.15, 0.8, 0.8])
    X1, y1 = get_y(data1)
    X2, y2 = get_y(data2)
    ax.errorbar(
        x=X1,
        y=np.array(y1["NegBinary"]) / np.array(y2["NegBinary"]),
        label="0.1/0.08 NegBin",
    )
    ax.errorbar(
        x=X1,
        y=np.array(y1["Constant"]) / np.array(y2["Constant"]),
        label="0.1/0.08 Const",
    )
    ax.errorbar(
        x=X1, y=np.array(y1["Binary"]) / np.array(y2["Binary"]), label="0.1/0.08 Bin"
    )
    ax.axhline(y=1)
    ax.legend()
    plt.draw()
    plt.savefig("./scaling_comparison_0p1_0p08.pdf")
    plt.show()


if __name__ == "__main__":
    # data = getdata()
    # plot(data)
    data1 = getdata_full(-0.05, 0.1)
    data2 = getdata_full(-0.04, 0.08)
    plot_compare(data1, data2)

    # data = getdata_full(-0.05, 0.1)
    plot_full(data1, "0p1")
    plot_full(data2, "0p08")
