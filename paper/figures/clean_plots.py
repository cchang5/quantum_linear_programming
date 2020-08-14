import matplotlib.pyplot as plt
from qlpdb.data.models import Data as data_Data
import pickle


def getallspinconfig(anneal_time=1, offset_tag="FixEmbedding_Single_Sided_Binary_-0.05_z1"):
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
    return spin_energy_count

def remap_basis(spin):
    # remap y
    remap = [1, 2, 3, 0, 1]
    for at in y.keys():
        X = list(y[at].keys())
        for v in X:
            temp = []
            for yi in y[at][v]:
                temp.append([yi[idx] for idx in remap])
            y[at][v] = np.array(temp)
    basis = [(i, j, k, l, m) for i in [0, 1] for j in [0, 1] for k in [0, 1] for l in [0, 1] for m in [0, 1]]
    scount = dict()
    for at in y.keys():
        X = list(y[at].keys())
        scount[at] = dict()
        for v in X:
            scount[at][v] = {b: 0 for b in basis}
            for yi in y[at][v]:
                scount[at][v][tuple(yi)] += 1

def plot_distribution():
    countm5 = getallspinconfig(offset_tag="FixEmbedding_Single_Sided_Binary_-0.05_z1")
    countp0 = getallspinconfig(offset_tag="FixEmbedding_Single_Sided_Binary_0.0_z1")
    countp5 = getallspinconfig(offset_tag="FixEmbedding_Single_Sided_Binary_0.05_z1")


plot_distribution()