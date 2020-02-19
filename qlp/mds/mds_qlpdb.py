import numpy as np
import hashlib
from qlpdb.graph.models import Graph as graph_Graph
from qlpdb.experiment.models import Experiment as experiment_Experiment
from qlpdb.data.models import Data as data_Data

def insert_result(graph_params, experiment_params, data_params):
    # select or insert row in graph
    graph, created = graph_Graph.objects.get_or_create(
	    tag=graph_params["tag"], # Tag for graph type (e.g. Hamming(n,m) or K(n,m))
    	total_vertices=graph_params["total_vertices"], # Total number of vertices in graph
	    total_edges=graph_params["total_edges"], # Total number of edges in graph
    	max_edges=graph_params["max_edges"], # Maximum number of edges per vertex
	    adjacency=graph_params["adjacency"], # Sorted adjacency matrix of dimension [N, 2]
    	adjacency_hash=graph_params["adjacency_hash"], # md5 hash of adjacency list used for unique constraint
    )

    # select or insert row in experiment
    experiment, created = experiment_Experiment.objects.get_or_create(
        graph=graph,  # Foreign Key to `graph`
        machine=experiment_params["machine"],  # Hardware name (e.g. DW_2000Q_5)
        settings=experiment_params["settings"],  # Store DWave machine parameters
        settings_hash=experiment_params["settings_hash"],  # md5 hash of key sorted settings
        p=experiment_params["p"],  # Coefficient of penalty term, 0 to 9999.99
        qubo=experiment_params["qubo"],  # Input QUBO to DWave
    )

    # select or insert row in data
    """
    data_data, created = data_Data.objects.get_or_create(
        experiment=experiment,  # Foreign Key to `experiment`
        measurement=,  # Increasing integer field labeling measurement number
        spin_config=,  # Spin configuration of solution, limited to 0, 1
        energy=,  # Energy corresponding to spin_config and QUBO
    )
    """
    return 0

def graph_summary(tag, graph):
    """
    Get summary statistics of input graph
    :param graph:
    :return:
    """
    vertices = np.unique(np.array([i for i in graph]).flatten())
    neighbors = {v: 0 for v in vertices}
    for i in graph:
        neighbors[i[0]] += 1
        neighbors[i[1]] += 1
    params = dict()
    params["tag"] = tag
    params["total_vertices"] = len(vertices)
    params["total_edges"] = len(graph)
    params["max_edges"] = max(neighbors)
    params["adjacency"] = [list(i) for i in list(graph)]
    params["adjacency_hash"] = hashlib.md5(
        str(np.sort(list(graph))).replace(" ", "").encode("utf-8")
    ).hexdigest()
    return params


def experiment_summary(machine, settings, penalty, qubo):
    params = dict()
    params["machine"] = machine
    params["settings"] = settings
    params["setting_hash"] = hashlib.md5(
        str([[key, settings[key]] for key in sorted(settings)])
        .replace(" ", "")
        .encode("utf-8")
    ).hexdigest()
    params["p"] = penalty
    params["qubo"] = qubo.todense().tolist()
    return params

def data_summary():
    pass
