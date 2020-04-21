from qlpdb.graph.models import Graph
from qlpdb.tdse.models import Tdse

from qlp.mds.mds_qlpdb import graph_summary
from qlp.mds import graph_tools as gt
from qlp.mds.qubo import get_mds_qubo


def parameters():
    # offset params
    offset_params = dict()
    offset_params["annealing_time"] = 0.0003
    offset_params["normalized_time"] = [-0.2, 1.2]
    offset_params["offset"] = "negbinary"
    offset_params["offset_min"] = -0.05
    offset_params["offset_range"] = 0.1
    offset_params["fill_value"] = "truncate"
    offset_params["anneal_curve"] = "linear"

    # wave params
    wave_params = dict()
    wave_params["type"] = "mixed"
    wave_params["temp"] = 15e-3
    wave_params["initial_wavefunction"] = "true"

    # graph params
    nvertices = 3
    graph, tag = gt.generate_nn_graph(nvertices)
    directed = False
    penalty = 2
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


def get_data():
    graph = Graph.objects.filter(tag="NN(3)").first()
    params = parameters()
    Tdse.objects.filter(offset_contains=params["offset"])
