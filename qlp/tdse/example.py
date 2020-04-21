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
        """This is a NN(2) graph embedded in Chimera"""
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
    elif n_vertices == 3:
        """This is a NN(3) graph embedded in Chimera"""
        q = """1120 1120 20.0
1121 1121 2.5
1122 1122 6.0
1123 1123 2.5
1124 1124 10.5
1125 1125 12.0
1126 1126 3.666666666666666
1127 1127 10.5
1128 1128 9.666666666666666
1132 1132 6.0
1134 1134 13.666666666666666
1135 1135 6.0
1120 1124 -4.0
1121 1124 2.0
1122 1124 -4.0
1123 1124 -16.0
1120 1125 -16.0
1121 1125 -4.0
1122 1125 8.0
1123 1125 -4.0
1120 1126 -8.0
1121 1126 8.0
1122 1126 -4.0
1123 1126 8.0
1120 1127 -4.0
1121 1127 -16.0
1122 1127 -4.0
1123 1127 2.0
1124 1132 -4.0
1128 1132 -4.0
1126 1134 -16.0
1128 1134 -16.0
1127 1135 -4.0
1128 1135 -4.0"""
        embedding = dict()
        embedding[0] = [1127, 1121]
        embedding[1] = [1134, 1126, 1128]
        embedding[2] = [1124, 1123]
        embedding[3] = [1135]
        embedding[4] = [1122]
        embedding[5] = [1120, 1125]
        embedding[6] = [1132]
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
