"""Example scripts for using the tdse module
"""
from typing import List, Dict, Tuple

from numpy import unique, array, concatenate
from scipy.sparse import dok_matrix


def embed_qubo_example(n_vertices: int) -> Tuple[dok_matrix, Dict[int, List[int]]]:
    """Returns qubo and explicit embedding for NN(n_vertices) graph

    Only implemented for n_vertices=2
    """
    if n_vertices == 2:
        # This is a NN(2) graph embedded in Chimera"""
        q = """640 640 2.5
641 641 6.0
643 643 6.0
645 645 -3.0
647 647 10.5
640 645 8.0
641 645 -4.0
643 645 -4.0
640 647 -16.0
641 647 -4.0
643 647 -4.0"""
        embedding = {0: [645], 1: [647, 640], 2: [641], 3: [643]}
    else:
        raise ValueError("No embedded graph defined.")

    q = array([[float(i) for i in qn.split(" ")] for qn in q.split("\n")])

    remap = {
        key: idx
        for idx, key in enumerate(unique(concatenate((q[:, 0], q[:, 1]), axis=0)))
    }
    qubo = dok_matrix((len(unique(q[:, 0])), len(unique(q[:, 0]))), dtype=float)
    for qi in q:
        i = remap[qi[0]]
        j = remap[qi[1]]
        qubo[i, j] = qi[2]
    return qubo, embedding
