import matplotlib.pyplot as plt
import pickle

import numpy as np

from qlpdb.tdse.models import Tdse

from qlp.mds.mds_qlpdb import graph_summary
from qlp.mds import graph_tools as gt
from qlp.mds.qubo import get_mds_qubo
from qlp.tdse import convert_params, embed_qubo_example

from django.conf import settings

label_params = dict()
label_params["fontsize"] = 7
tick_params = dict()
tick_params["labelsize"] = 7
tick_params["width"] = 0.5
errorbar_params = dict()
errorbar_params["mew"] = 0.5
errorbar_params["markersize"] = 5
red = "#c82506"
green = "#70b741"
blue = "#51a7f9"


class Data:
    def __init__(self):
        self.params = self.parameters()

    def parameters(self):
        # offset params
        offset_params = dict()
        offset_params["annealing_time"] = 1
        offset_params["normalized_time"] = [0, 1]
        offset_params["offset"] = "binary"
        offset_params["offset_min"] = 0
        offset_params["offset_range"] = 0
        offset_params["fill_value"] = "extrapolate"
        offset_params["anneal_curve"] = "dwave"

        # wave params
        wave_params = dict()
        wave_params["type"] = "mixed"
        wave_params["temp"] = 15e-3
        wave_params["gamma"] = 0.05
        wave_params["initial_wavefunction"] = "transverse"

        # graph params
        nvertices = 2
        graph, tag = gt.generate_nn_graph(nvertices)
        directed = False
        penalty = 2
        embed = True  # nvertices = [2, 3] available

        if embed:
            qubo, embedding = embed_qubo_example(nvertices)
        else:
            qubo = get_mds_qubo(
                graph, directed=directed, penalty=penalty, triangularize=True, dtype="d"
            )
        graph_params = graph_summary(tag, graph, qubo)

        # solver params
        solver_params = dict()
        solver_params["method"] = "RK45"
        solver_params["rtol"] = 1e-6
        solver_params["atol"] = 1e-7

        params = {
            "offset": offset_params,
            "wave": wave_params,
            "graph": graph_params,
            "solver": solver_params,
        }
        return params

    def get_data(self):
        query = Tdse.objects.filter(
            graph__tag=self.params["graph"]["tag"],
            offset__contains=convert_params(self.params["offset"]),
            solver__contains=self.params["solver"],
            wave__contains=self.params["wave"],
        ).first()
        return query


def aggregate():
    adata = dict()
    data = Data()
    #for offset in [
    #    0.05,
    #    0.04,
    #    0.03,
    #    0.02,
    #    0.01,
    #    0.0,
    #    -0.0005,
    #    -0.001,
    #    -0.002,
    #    -0.004,
    #    -0.005,
    #    -0.01,
    #    -0.02,
    #    -0.03,
    #    -0.04,
    #    -0.05,
    #]:
        # for offset in [-0.04, 0.04]:
    for offset in [-0.04, 0.04]:
        data.params["offset"]["offset"] = "binary"
        data.params["offset"]["offset_min"] = offset
        data.params["offset"]["offset_range"] = abs(offset) * 2
        adata[offset] = data.get_data()
        with open(f"{settings.MEDIA_ROOT}/{adata[offset].solution}", "rb") as file:
            adata[offset].sol = pickle.load(file)
        with open(f"{settings.MEDIA_ROOT}/{adata[offset].instance}", "rb") as file:
            adata[offset].tdse = pickle.load(file)
    return adata


def aggregate_nodeco():
    adata = dict()
    data = Data()
    data.params["wave"]["gamma"] = 0.0
    data.params["offset"]["offset"] = "binary"
    #for offset in [
    #    0.05,
    #    0.04,
    #    0.03,
    #    0.02,
    #    0.01,
    #    0.0,
    #    -0.01,
    #    -0.02,
    #    -0.03,
    #    -0.04,
    #    -0.05,
    #]:
    for offset in [0.04, -0.04]:
        data.params["offset"]["offset_min"] = offset
        data.params["offset"]["offset_range"] = abs(offset) * 2
        adata[offset] = data.get_data()
        with open(f"{settings.MEDIA_ROOT}/{adata[offset].solution}", "rb") as file:
            adata[offset].sol = pickle.load(file)
        with open(f"{settings.MEDIA_ROOT}/{adata[offset].instance}", "rb") as file:
            adata[offset].tdse = pickle.load(file)
    return adata


def aggregate_gamma():
    adata = dict()
    data = Data()
    data.params["offset"]["offset"] = "binary"
    data.params["wave"]["temp"] = 15e-3
    for gamma in [
        0.001,
        0.01,
        0.02,
        0.03,
        0.035,
        0.04,
        0.045,
        0.05,
        0.055,
        0.06,
        0.065,
        0.07,
        0.08,
        0.09,
        0.1,
    ]:
        print(gamma)
        data.params["wave"]["gamma"] = gamma
        adata[gamma] = data.get_data()
        with open(f"{settings.MEDIA_ROOT}/{adata[gamma].solution}", "rb") as file:
            adata[gamma].sol = pickle.load(file)
        with open(f"{settings.MEDIA_ROOT}/{adata[gamma].instance}", "rb") as file:
            adata[gamma].tdse = pickle.load(file)
    return adata


def plot_aggregate(adata, tag):
    plt.figure("full probability", figsize=(7, 4))
    ax = plt.axes([0.15, 0.15, 0.8, 0.8])
    for idx, key in enumerate(adata):
        if key > 0:
            color = red
        else:
            color = blue
        tdse = adata[key].tdse
        idx, en, evec = tdse.ground_state_degeneracy(tdse.IsingH, 2e-2, debug=False)
        ax.errorbar(x=adata[key].time, y=adata[key].prob, color=color, label=f"{int(key*2*100)}%")
    ax.set_xlabel("normalized time")
    ax.set_ylabel("MDS probability")
    ax.legend()
    plt.draw()
    plt.savefig(f"full_prob_{tag}.pdf", transparent=True)
    """
    plt.figure("prob", figsize=(7, 4))
    ax = plt.axes([0.15, 0.15, 0.8, 0.8])
    x = list(adata.keys())
    y = [adata[key].prob[-1] for key in x]
    for idx, xi in enumerate(x):
        if float(xi) < 0 and float(xi) > -0.01:
            color = red
        else:
            color = blue
        ax.errorbar(x=2*xi, y=y[idx], ls="none", marker="o", color=color)
    ax.set_xlabel("offset range (%)")
    ax.set_ylabel("MDS probability")
    if tag == "deco":
        ax.set_ylim([0.85, 0.92])
    else:
        ax.set_ylim([0.9938, 0.99485])
    plt.draw()
    plt.savefig(f"./sim_{tag}.pdf", transparent=True)
    """
    reg = 1e-9
    plt.figure(f"mutual information", figsize=(7, 4))
    ax = plt.axes([0.15, 0.15, 0.8, 0.8])
    mi = dict()
    for idx, key in enumerate(adata):
        if key > 0:
            color = red
        else:
            color = blue
        sol = adata[key].sol
        tdse = adata[key].tdse
        entropy_params = {"nA": 2, "indicesA": "abcdfabceg->dfeg", "reg": reg}
        entropyA = np.asarray(
            [tdse.ent_entropy(sol.y[:, i], **entropy_params) for i in range(sol.t.size)]
        ).real
        entropy_params = {"nA": 3, "indicesA": "aceghbdfgh->acebdf", "reg": reg}
        entropyB = np.asarray(
            [tdse.ent_entropy(sol.y[:, i], **entropy_params) for i in range(sol.t.size)]
        ).real
        entropyAB = np.asarray(
            [tdse.vonNeumann_entropy(sol.y[:, i], reg) for i in range(sol.t.size)]
        ).real
        mutual_information = entropyA + entropyB - entropyAB
        ax.errorbar(x=adata[key].time, y=mutual_information, color=color, label=f"{int(key*2*100)}%")
        mi[key] = mutual_information
    ax.set_xlabel("normalized time")
    ax.set_ylabel("mutual information")
    ax.legend()
    plt.draw()
    plt.savefig(f"mutual_info_{tag}.pdf", transparent=True)

    plt.figure(f"final mutual information", figsize=(7, 4))
    ax = plt.axes([0.15, 0.15, 0.8, 0.8])
    x = list(mi.keys())
    y = [mi[key][-1] for key in x]
    ax.errorbar(x=x, y=y, ls="None", marker="o")
    ax.set_xlabel("offset")
    plt.show()


def plot_gamma(gdata):
    plt.figure("decoherence")
    ax = plt.axes([0.15, 0.15, 0.8, 0.8])
    x = list(gdata.keys())
    X = 1 / (np.array(x) * 1e9) * 1e9
    y = [gdata[key].prob[-1] for key in x]
    ax.errorbar(x=X, y=y, ls="none", marker="o")
    ax.set_xlabel("T1 (ns)")
    plt.draw()
    plt.show()

    plt.figure("decoherence full")
    ax = plt.axes([0.15, 0.15, 0.8, 0.8])
    t = list(gdata.keys())
    T1 = 1 / (np.array(t) * 1e9) * 1e9
    for idx, key in enumerate(t):
        x = gdata[key].time
        y = gdata[key].prob
        ax.errorbar(x=x, y=y, ls="-", label=T1[idx])
    ax.set_xlabel("normalized time")
    ax.legend()
    plt.draw()
    plt.show()


def plot_schedule(adata):
    X, y1, y2 = adata[-0.05].tdse.AS.plot([0, 1])
    X, z1, z2 = adata[0.0].tdse.AS.plot([0, 1])

    plt.figure("anneal schedule", figsize=(7, 4))
    ax = plt.axes([0.15, 0.15, 0.8, 0.8])
    ax.errorbar(x=X, y=y1[:, 0], ls="-", color=blue, label="Positive offset A(s)")
    ax.errorbar(x=X, y=z1[:, 0], ls="-", color="k", label="Default A(s)")
    ax.errorbar(x=X, y=y1[:, -1], ls="-", color=red, label="Negative offset A(s)")
    ax.errorbar(x=X, y=y2[:, 0], ls="--", color=blue, label="Positive offset B(s)")
    ax.errorbar(x=X, y=z2[:, 0], ls="--", color="k", label="Default B(s)")
    ax.errorbar(x=X, y=y2[:, -1], ls="--", color=red, label="Negative offset B(s)")
    ax.set_xlabel("normalized time")  # , **label_params)
    ax.set_ylabel("GHz")  # , **label_params)
    ax.legend()
    plt.draw()
    plt.savefig("./anneal_schedule.pdf")
    plt.show()


def plot_spectrum(adata):
    from numpy.linalg import eigh

    # plot spectrum
    tdse = adata[0.0].tdse
    normalized_time = tdse.offset["normalized_time"]

    num_es = 5
    # unit conversion
    # sol.y = [GHz / h]
    energyscale = tdse.ising["energyscale"]
    make_dimensionless = 1 / energyscale * 1 / tdse.AS.B(normalized_time[1])[0]
    print("ENERGY SCALE:", energyscale)
    fig = plt.figure("spectrum", figsize=(7, 4))
    ax = plt.axes([0.15, 0.15, 0.8, 0.8])

    x = []
    y = {es: [] for es in range(num_es)}
    for s in np.linspace(normalized_time[0], normalized_time[1]):
        eigval, eigvec = eigh((tdse.annealingH(s)).toarray())
        seigval = (
            np.sort(eigval) * make_dimensionless
            + tdse.ising["c"]
            + tdse.ising["qubo_constant"]
        )
        x.append(s)
        for es in y.keys():
            y[es].append(seigval[es])
    for es in y.keys():
        ax.errorbar(x=x, y=y[es])
    gap = {es: np.array(y[es]) - np.array(y[0]) for es in y.keys()}
    ax.set_xlabel("normalized time")
    ax.set_ylabel("energy (dimensionless)")
    plt.draw()
    plt.savefig("./spectrum.pdf", transparent=True)
    plt.show()
    print("ground state energy:", y[0][-1])
    print("1st ex state energy:", y[1][-1])
    eigval, eigvec = eigh((tdse.annealingH(s)).toarray())
    print("ground state eigvec:", np.round(eigvec[0]))
    print("1st ex state eigvec:", np.round(eigvec[1]))
    kb = 8.617333262145e-5  # eV⋅K−1
    h = 4.135667696e-15  # eV⋅s
    h_kb = h / kb  # K*s
    print(
        "start gap energy (Kelvins):",
        energyscale
        * tdse.AS.B(normalized_time[1])[0]
        * (y[1][0] - y[0][0])
        * 1e9
        * h_kb,
    )
    mingap = min(np.array(y[2]) - np.array(y[0]))
    print("MINGAP:", mingap)
    print("min gap energy (Kelvins):", mingap / make_dimensionless * 1e9 * h_kb)

    # plot spectrum
    tdse = adata[-0.05].tdse
    normalized_time = tdse.offset["normalized_time"]

    num_es = 5
    # unit conversion
    # sol.y = [GHz / h]
    energyscale = tdse.ising["energyscale"]
    make_dimensionless = 1 / energyscale * 1 / tdse.AS.B(normalized_time[1])[0]

    fig = plt.figure("offset spectrum", figsize=(7, 4))
    ax = plt.axes([0.15, 0.15, 0.8, 0.8])

    x = []
    y = {es: [] for es in range(num_es)}
    for s in np.linspace(normalized_time[0], normalized_time[1]):
        eigval, eigvec = eigh((tdse.annealingH(s)).toarray())
        seigval = (
            np.sort(eigval) * make_dimensionless
            + tdse.ising["c"]
            + tdse.ising["qubo_constant"]
        )
        x.append(s)
        for es in y.keys():
            y[es].append(seigval[es])
    for es in y.keys():
        ax.errorbar(x=x, y=y[es])
    gap = {es: np.array(y[es]) - np.array(y[0]) for es in y.keys()}
    ax.set_xlabel("normalized time")
    ax.set_ylabel("energy (dimensionless)")
    plt.draw()
    plt.savefig("./spectrum_offset.pdf", transparent=True)
    plt.show()
    print("ground state energy:", y[0][-1])
    print("1st ex state energy:", y[1][-1])
    eigval, eigvec = eigh((tdse.annealingH(s)).toarray())
    print("ground state eigvec:", np.round(eigvec[0]))
    print("1st ex state eigvec:", np.round(eigvec[1]))
    kb = 8.617333262145e-5  # eV⋅K−1
    h = 4.135667696e-15  # eV⋅s
    h_kb = h / kb  # K*s
    print(
        "start gap energy (Kelvins):",
        energyscale
        * tdse.AS.B(normalized_time[1])[0]
        * (y[1][0] - y[0][0])
        * 1e9
        * h_kb,
    )
    mingap = min(np.array(y[2]) - np.array(y[0]))
    print(
        "min gap energy (Kelvins):",
        energyscale * tdse.AS.B(normalized_time[1])[0] * mingap * 1e9 * h_kb,
    )


if __name__ == "__main__":
    # vs offset
    adata = aggregate()
    # print(list(adata.keys()))
    plot_aggregate(adata, "deco")

    # plot AS
    # print(list(adata.keys()))
    # plot_schedule(adata)

    # plot spectrum
    # plot_spectrum(adata)

    # vs offset no decoherence
    bdata = aggregate_nodeco()
    plot_aggregate(bdata, "nodeco")

    # vs gamma
    # gdata = aggregate_gamma()
    # plot_gamma(gdata)
