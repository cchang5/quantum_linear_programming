import matplotlib.pyplot as plt
from qlpdb.data.models import Data as data_Data
import numpy as np


def single_histogram(datadict):
    """Plots histogram of results
    """
    fig, axs = plt.subplots(
        len(datadict), 1, sharey=True, sharex=True, tight_layout=True, figsize=(5, 15)
    )
    print(np.shape(axs))
    for idx, key in enumerate(datadict):
        query = datadict[key].groupby("energy").count()["id"]
        X = query.index
        height = np.log(query.values)
        axs[idx].bar(x=X, height=height, width=0.9, bottom=0, align="center")
    plt.draw()
    plt.show()

def degenfcn(n):
    if n%3 == 2:
        return n//3+2
    if n%3 == 1:
        return 2*(n//3+1)
    if n%3 == 0:
        return 1

def plot_prob(datadict):
    """Plots percentage getting correct answer
    """
    X = [datadict[key]["energy"].min() for key in datadict.keys()]
    X = range(2, len(datadict.keys())+2)
    y = [
        datadict[key].groupby("energy").count()["id"].iloc[0]
        / datadict[key].count()["id"]
        for key in datadict.keys()
    ]
    degeneracy = np.array([degenfcn(key) for key in datadict.keys()])
    yscaled = y/degeneracy
    rguess = np.array([1/2**xi for xi in X])*degeneracy

    fig = plt.figure(figsize=(7, 4))
    ax = plt.axes([0.15, 0.15, 0.8, 0.8])
    ax.errorbar(x=X, y=y, marker="o", label="Exp. result")
    #ax.errorbar(x=X, y=yscaled, marker="s", label="Exp. scaled with degeneracy")
    ax.errorbar(x=X, y=rguess, marker="^", label="Random guessing")
    ax.errorbar(x=X, y=y/rguess, marker="*", label="Exp. result / Random guessing")
    ax.set_yscale("log")
    ax.set_xlabel("Nodes in graph")
    ax.set_ylabel("Probability of getting MDS solution")
    plt.legend()
    plt.savefig("./scaling_plot.pdf")
    plt.draw()
    plt.show()


if __name__ == "__main__":
    nndata = {
        key: data_Data.objects.filter(
            experiment__graph__tag=f"NN({key})"
        ).filter(experiment__percentage=f"{pct}").to_dataframe()
        for key in range(2, 15) for pct in [0, 0.05]
    }

    #single_histogram(nndata)
    plot_prob(nndata)
