import matplotlib.pyplot as plt
from qlpdb.data.models import Data as data_Data
import pickle


def getallspinconfig(anneal_time=1, offset_tag="FixEmbedding_Binary_-0.05_0.1_v5"):
    data = data_Data.objects.filter(
        experiment__graph__tag=f"NN(2)",
        experiment__tag=offset_tag,
        experiment__settings__annealing_time=anneal_time,
        experiment__settings__num_spin_reversal_transforms=5,
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
    print(spin_energy_count)
getallspinconfig()
