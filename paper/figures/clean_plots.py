import matplotlib.pyplot as plt
from qlpdb.data.models import Data as data_Data
import pickle
import numpy as np

from django.conf import settings
from qlpdb.tdse.models import Tdse
from qlp.tdse import convert_params, embed_qubo_example
from qlp.mds import graph_tools as gt
from qlp.mds.mds_qlpdb import graph_summary


figsize = (7, 4)
ratio = [0.15, 0.15, 0.8, 0.8]
"""
###################################
#####  Simulation Data Class  #####
###################################
"""
class Sim:
    def __init__(self):
        self.params = self.parameters()

    def parameters(self):
        # offset params
        offset_params = dict()
        offset_params["annealing_time"] = 1
        offset_params["offset"] = "single_sided_binary"
        offset_params["offset_min"] = 0
        offset_params["fill_value"] = "extrapolate"
        offset_params["anneal_curve"] = "dwave"

        # wave params
        wave_params = dict()
        wave_params["type"] = "mixed"
        wave_params["temp"] = 0.02
        wave_params["gamma"] = 1 / 2
        wave_params["initial_wavefunction"] = "transverse"

        # graph params
        nvertices = 2
        graph, tag = gt.generate_nn_graph(nvertices)
        qubo, embedding = embed_qubo_example(nvertices)
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

    def get_data(self, offset):
        self.params["offset"]["offset_min"] = offset
        print(self.params["graph"]["tag"])
        print(convert_params(self.params["offset"]))
        print(self.params["solver"])
        print(self.params["wave"])
        query = Tdse.objects.filter(
            graph__tag=self.params["graph"]["tag"],
            offset__contains=convert_params(self.params["offset"]),
            solver__contains=self.params["solver"],
            wave__contains=self.params["wave"],
        ).first()
        with open(f"{settings.MEDIA_ROOT}/{query.solution}", "rb") as file:
            sol = pickle.load(file)
        with open(f"{settings.MEDIA_ROOT}/{query.instance}", "rb") as file:
            tdse = pickle.load(file)
        return query, sol, tdse

"""
###########################################
#####  Plot final state distribution  #####
###########################################
"""


def getallspinconfig(anneal_time=1, offset_tag="FixEmbedding_Single_Sided_Binary_-0.05_z3"):
    data = data_Data.objects.filter(
        experiment__graph__tag=f"NN(2)",
        experiment__tag=offset_tag,
        experiment__settings__annealing_time=anneal_time,
        experiment__settings__num_spin_reversal_transforms=0,
    ).to_dataframe()

    spin = data["spin_config"]
    energy = data["energy"]
    spin_energy_count = dict()
    for idx in range(len(energy)):
        tag = (tuple(spin[idx]), energy[idx])
        if tag in spin_energy_count.keys():
            spin_energy_count[tag] += 1
        else:
            spin_energy_count[tag] = 1
    total = sum([spin_energy_count[tag] for tag in spin_energy_count])
    spin_energy_count = {tag: spin_energy_count[tag] / total for tag in spin_energy_count}
    return spin_energy_count


def plot_distribution():
    offset = 0.01 * (np.arange(11) - 5)
    count_energy = {oi: getallspinconfig(offset_tag=f"FixEmbedding_Single_Sided_Binary_{oi}_z3") for oi in offset}
    count = {oi: {spin_energy[0]: count_energy[oi][spin_energy] for spin_energy in count_energy[oi]} for oi in
             count_energy}
    energy = {oi: {spin_energy[0]: spin_energy[1] for spin_energy in count_energy[oi]} for oi in count_energy}
    print(count)
    # Xlabel = [tuple([int(q) for q in list(format(i, '05b'))]) for i in range(2 ** 5)]
    # X = np.arange(2 ** 5)
    Xlabel = [(0, 1, 0, 0, 0), (1, 0, 0, 1, 0), (1, 1, 1, 1, 1)]
    X = np.array(range(len(Xlabel)))

    plt.figure(figsize=figsize)
    ax = plt.axes(ratio)
    for idx, os in enumerate(offset):
        height = []
        for Xi in Xlabel:
            if Xi in count[os]:
                height.append(count[os][Xi])
            else:
                height.append(0)
        ax.bar(x=X + idx / len(offset), height=height, width=1 / len(offset), alpha=0.5, label=os)

    ax.set_xticks(X)
    ax.set_xticklabels(Xlabel, rotation="90")

    plt.legend()
    plt.draw()
    plt.show()


"""
######################################
#####  plot prob vs anneal time  #####
######################################
"""


def getallannealtime(anneal_time=1, offset_tag="FixEmbedding_Single_Sided_Binary_0_z3"):
    data = data_Data.objects.filter(
        experiment__graph__tag=f"NN(6)",
        experiment__tag=offset_tag,
        experiment__settings__annealing_time=anneal_time,
        experiment__settings__num_spin_reversal_transforms=0,
    ).to_dataframe()

    ground_state_count = data.groupby("energy").count()["id"].sort_index().iloc[0]
    total_count = data.count()["id"]
    prob = ground_state_count / total_count
    print(anneal_time, prob)
    return prob


def plot_anneal_time():
    anneal_time = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700,
                   800, 900, 1000, 2000]
    prob = [getallannealtime(anneal_time=at) for at in anneal_time]

    plt.figure(figsize=figsize)
    ax = plt.axes(ratio)

    ax.errorbar(x=anneal_time, y=prob, ls="-", marker="o", color="k")

    ax.set_xscale("log")

    plt.draw()
    plt.show()


"""
################################################
#####  plot prob vs offset and graph size  #####
################################################
"""


def getall(offset=-0.05, graphsize=2):
    """
    z3 uses qubits with offset range from -0.05 to 0.05
    z4 uses qubits with offset range from -0.08 to 0.03
    """
    if True:
        prob = {
            -0.05: [0.943775, 0.3779625, 0.7455375, 0.2804, 0.1375625, 0.174, 0.019525, 0.0149125, 0.089575, 0.0175875],
            -0.04: [0.942, 0.4114625, 0.7426875, 0.3423, 0.1701125, 0.23095714285714286, 0.0314125, 0.025775, 0.103325,
                    0.024375],
            -0.03: [0.9414375, 0.4112125, 0.7368875, 0.4222875, 0.2222, 0.2960375, 0.0439875, 0.01975, 0.142625,
                    0.0323875],
            -0.02: [0.9461375, 0.38595, 0.7510875, 0.4678875, 0.2430875, 0.3330625, 0.05745, 0.020325, 0.1754,
                    0.041575],
            -0.01: [0.9559, 0.38915, 0.7507875, 0.59275, 0.304275, 0.4103375, 0.0728625, 0.0317, 0.1949, 0.0475875],
            0.0: [0.95615, 0.41495, 0.7429875, 0.657625, 0.3045875, 0.436725, 0.084625, 0.0288625, 0.22375, 0.0555875],
            0.01: [0.962475, 0.4271, 0.749475, 0.731925, 0.364775, 0.4652375, 0.0925375, 0.0458875, 0.244, 0.0653125],
            0.02: [0.9637375, 0.4123875, 0.7577875, 0.767425, 0.3957125, 0.4703875, 0.0889102564102564, 0.0537125,
                   0.2728625, 0.0788875],
            0.03: [0.97095, 0.4319375, 0.7673875, 0.8057625, 0.4305875, 0.497775, 0.095, 0.054175, 0.2614125,
                   0.0884375],
            0.04: [0.9704375, 0.4289625, 0.7738, 0.8369625, 0.4561875, 0.4599875, 0.0942, 0.052275, 0.2599625,
                   0.0871625],
            0.05: [0.9730875, 0.49125, 0.7733, 0.861175, 0.4954375, 0.445675, 0.10225, 0.0489125, 0.2658125, 0.0861875]}

        if len(prob[offset]) + 1 >= graphsize:
            return prob[offset][graphsize - 2]
        else:
            pass
    data = data_Data.objects.filter(
        experiment__graph__tag=f"NN({graphsize})",
        experiment__tag=f"FixEmbedding_Single_Sided_Binary_{offset}_z4",
        experiment__settings__annealing_time=500,
        experiment__settings__num_spin_reversal_transforms=0,
    ).to_dataframe()
    print(data.groupby("energy").count()["id"].sort_index())
    ground_state_count = data.groupby("energy").count()["id"].sort_index().iloc[0]
    total_count = data.count()["id"]
    prob = ground_state_count / total_count
    print(offset, f"NN({graphsize})", prob, total_count)
    return prob


def plot_all():
    offset = list(0.01 * (np.arange(11) - 5))
    graphsize = list(np.arange(2, 12))
    prob = {os: [getall(os, gs) for gs in graphsize] for os in offset}

    plt.figure(figsize=figsize)
    ax = plt.axes(ratio)

    for os in offset:
        ax.errorbar(x=graphsize, y=prob[os], ls="-", marker="o", label=os)

    plt.legend()

    print(prob)

    plt.draw()
    plt.show()


"""
######################################
#####  data for tdse simulation  #####
######################################
"""
def gettdse(offset=0.0):
    if True:
        prob = {-0.07: 0.70107, -0.06: 0.72948, -0.05: 0.79639, -0.04: 0.82127, -0.03: 0.84074, -0.02: 0.84374,
                -0.01: 0.88497, 0.0: 0.90447, 0.01: 0.9189, 0.02: 0.92944, 0.03: 0.94336, 0.04: 0.94864, 0.05: 0.95251,
                0.06: 0.93402, 0.07: 0.92907}
        if offset in prob.keys():
            return prob[offset]
        else:
            pass
    data = data_Data.objects.filter(
        experiment__graph__tag=f"NN(2)",
        experiment__tag=f"FixEmbedding_Single_Sided_Binary_{offset}_z3",
        experiment__settings__annealing_time=1,
        experiment__settings__num_spin_reversal_transforms=0,
    ).to_dataframe()
    print(offset)
    energy_count = data.groupby("energy").count()["id"].sort_index().iloc[0]
    total_count = data.count()["id"]
    prob = energy_count / total_count
    print(total_count, prob)
    return prob

def gettdsetheory(offset=0.0):
    sim = Sim()
    query, sol, tdse = sim.get_data(offset)
    print(sol.time)
    print(sol.prob)

def plot_tdse():
    offset = list(0.01 * (np.arange(15) - 7))
    prob = [gettdse(os) for os in offset]
    plt.figure(figsize=figsize)
    ax = plt.axes(ratio)

    ax.errorbar(x=offset, y=prob, ls="-", marker="o")

    plt.draw()
    plt.show()


"""
##################
#####  main  #####
##################
"""
if __name__ == "__main__":
    """
    For DWave only
    """
    # plot_anneal_time()
    # plot_all()
    """
    For TDSE simulation
    """
    #plot_tdse()
    # plot_distribution()

    gettdsetheory(-0.05)
