import numpy as np
import pandas as pd
import hashlib
from qlpdb.graph.models import Graph as graph_Graph
from qlpdb.experiment.models import Experiment as experiment_Experiment
from qlpdb.data.models import Data as data_Data


def insert_result(graph_params, experiment_params, data_params):
    # select or insert row in graph
    graph, created = graph_Graph.objects.get_or_create(
        tag=graph_params["tag"],  # Tag for graph type (e.g. Hamming(n,m) or K(n,m))
        total_vertices=graph_params[
            "total_vertices"
        ],  # Total number of vertices in graph
        total_edges=graph_params["total_edges"],  # Total number of edges in graph
        max_edges=graph_params["max_edges"],  # Maximum number of edges per vertex
        adjacency=graph_params[
            "adjacency"
        ],  # Sorted adjacency matrix of dimension [N, 2]
        adjacency_hash=graph_params[
            "adjacency_hash"
        ],  # md5 hash of adjacency list used for unique constraint
    )

    # select or insert row in experiment
    experiment, created = experiment_Experiment.objects.get_or_create(
        graph=graph,  # Foreign Key to `graph`
        machine=experiment_params["machine"],  # Hardware name (e.g. DW_2000Q_5)
        settings=experiment_params["settings"],  # Store DWave machine parameters
        settings_hash=experiment_params[
            "settings_hash"
        ],  # md5 hash of key sorted settings
        p=experiment_params["p"],  # Coefficient of penalty term, 0 to 9999.99
        fact=experiment_params["fact"], # Manual rescale coefficient, float
        chain_strength=experiment_params["chain_strength"],
        qubo=experiment_params["qubo"],  # Input QUBO to DWave
    )

    # select or insert row in data
    for idx in range(len(data_params["spin_config"])):
        measurement = data_Data.objects.filter(experiment=experiment).order_by(
            "-measurement"
        )
        if measurement.exists():
            measurement = measurement.first().measurement + 1
        else:
            measurement = 0
        data, created = data_Data.objects.get_or_create(
            experiment=experiment,  # Foreign Key to `experiment`
            measurement=measurement,  # Increasing integer field labeling measurement number
            spin_config=list(
                data_params["spin_config"][idx]
            ),  # Spin configuration of solution, limited to 0, 1
            chain_break_fraction=data_params["chain_break_fraction"][idx],
            energy=data_params["energy"][
                idx
            ],  # Energy corresponding to spin_config and QUBO
            constraint_satisfaction=data_params["constraint_satisfaction"][idx],
        )
    return data


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
    params["max_edges"] = max(neighbors.values())
    params["adjacency"] = [list(i) for i in list(graph)]
    params["adjacency_hash"] = hashlib.md5(
        str(np.sort(list(graph))).replace(" ", "").encode("utf-8")
    ).hexdigest()
    return params


def experiment_summary(machine, settings, penalty, factor, chain_strength, qubo):
    params = dict()
    params["machine"] = machine
    params["settings"] = settings
    params["p"] = penalty
    params["fact"] = factor
    params["chain_strength"] = chain_strength
    norm_params = pd.io.json.json_normalize(params, sep="_").to_dict()
    norm_params = {key:norm_params[key][0] for key in norm_params}
    params["settings_hash"] = hashlib.md5(
        str([[key, norm_params[key]] for key in sorted(norm_params)])
        .replace(" ", "")
        .encode("utf-8")
    ).hexdigest()
    params["qubo"] = qubo.todense().tolist()
    return params


def data_summary(raw, graph_params, experiment_params):
    params = dict()
    params["spin_config"] = raw.iloc[:, : len(experiment_params["qubo"])].values
    params["energy"] = (
        raw["energy"].values * experiment_params["fact"]
        + experiment_params["p"] * graph_params["total_vertices"]
    )
    params["chain_break_fraction"] = raw["chain_break_fraction"].values
    params["constraint_satisfaction"] = np.equal(
        params["energy"],
        np.sum(params["spin_config"][:, : graph_params["total_vertices"]], axis=1),
    )
    return params


def QUBO_to_Ising(Q):
    q = np.diagonal(Q)
    QD = np.copy(Q)
    for i in range(len(QD)):
        QD[i, i] = 0.0
    QQ = np.copy(QD + np.transpose(QD))
    J = np.triu(QQ) / 4.0
    uno = np.ones(len(QQ))
    h = q / 2 + np.dot(QQ, uno) / 4
    g = np.dot(uno, np.dot(QD, uno)) / 4.0 + np.dot(q, uno) / 2.0
    return (J, h, g)
