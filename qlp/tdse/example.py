"""Example scripts for using the tdse module
"""
from typing import List, Dict, Tuple

from numpy import unique, array, concatenate, diagonal
from scipy.sparse import dok_matrix
from dimod import ising_to_qubo, qubo_to_ising


def embed_qubo_example(n_vertices: int) -> Tuple[dok_matrix, Dict[int, List[int]]]:
    """Returns qubo and explicit embedding for NN(n_vertices) graph

    Only implemented for n_vertices=2
    """
    if n_vertices == 2:
        """This is a NN(2) graph embedded in Chimera
        h = [-3.0, 10.5, 2.5, 6.0, 6.0]
        J = [[ 0  0  8 -4 -4],
             [ 0  0-16 -4 -4],
             [ 0  0  0  0  0],
             [ 0  0  0  0  0],
             [ 0  0  0  0  0]]
        """
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
#    if n_vertices == 2:
#        """NN(2) on 2000Q_6"""
#        q = """1890 1890 -0.75
#1891 1891 2.625
#1892 1892 0.625
#1893 1893 1.5
#1894 1894 1.5
#1890 1892 2.0
#1891 1892 -4.0
#1890 1893 -1.0
#1891 1893 -1.0
#1890 1894 -1.0
#1891 1894 -1.0"""
#    embedding = {0: [1892, 1891], 1: [1890], 2: [1893], 3: [1894]}
    elif n_vertices == 2.1:
        """by hand put in here
        2000Q_6 result
        embedding = {0: [1892, 1891], 1: [1890], 2: [1893], 3: [1894]}
        h = [0.375  0.375 -0.25 -0.25]
        J = [[0.    0.5   -0.25 -0.25]
             [0.    0.    -0.25 -0.25]
             [0.    0.     0.    0.  ]
             [0.    0.     0.    0.  ]]
        chain strength = 1.0
        1890 1890 -> 1 1
        1891 1891 -> 0 0
        1892 1892 -> 0 0
        1893 1893 -> 2 2
        1894 1894 -> 3 3
        1890 1892 -> 1 0
        1891 1892 -> 0 0 chain strength 
        1890 1893 -> 1 2
        1891 1893 -> 0 2
        1890 1894 -> 1 3
        1891 1894 -> 0 3
        """
        q = """1890 1890 0.375
1891 1891 0.375
1892 1892 0.375
1893 1893 -0.25
1894 1894 -0.25
1890 1892 0.5
1891 1892 -1.0
1890 1893 -0.25
1891 1893 -0.25
1890 1894 -0.25
1891 1894 -0.25"""
        embedding = {0: [1892, 1891], 1: [1890], 2: [1893], 3: [1894]}
    elif n_vertices == 2.2:
        """by hand put in here
        2000Q_5 result
        embedding = {0: [645], 1: [647, 640], 2: [641], 3: [643]}
        h = [0.375  0.375 -0.25 -0.25]
        J = [[0.    0.5   -0.25 -0.25]
             [0.    0.    -0.25 -0.25]
             [0.    0.     0.    0.  ]
             [0.    0.     0.    0.  ]]
        chain strength = 1.0
        640 640 -> 1 1
        641 641 -> 2 2
        643 643 -> 3 3
        645 645 -> 0 0
        647 647 -> 1 1
        640 645 -> 1 0
        641 645 -> 2 0
        643 645 -> 3 0
        640 647 -> 1 1 chain strength
        641 647 -> 2 1
        643 647 -> 3 1"""
        q = """640 640 0.375
641 641 -0.25
643 643 -0.25
645 645 0.375
647 647 0.375
640 645 0.5
641 645 -0.25
643 645 -0.25
640 647 -1.0
641 647 -0.25
643 647 -0.25"""
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
    h = dict()
    J = dict()
    for qi in q:
        i = remap[qi[0]]
        j = remap[qi[1]]
        if i == j:
            h[i] = qi[2]
        else:
            J[(i,j)] = qi[2]

    """NEED TO REPLACE THIS WITH OWN FUNCTION
    ising_to_qubo ->
    mds_qlpdb.Ising_to_QUBO(J, h)
    return dok_matrix
    
    from qlp.mds.mds_qlpdb import QUBO_to_Ising 
    Cross check with QUBO_to_Ising(dok_matrix.todense().tolist())
    to see if we get back same Ising that was put in.
    
    """
    from qlp.mds.mds_qlpdb import Ising_to_QUBO

    qubo = Ising_to_QUBO(J, h)  # currently h and J are dictionaries. you can change them to vector and matrix if you want
    dok_qubo = dok_matrix(qubo)

    from qlp.mds.mds_qlpdb import QUBO_to_Ising
    print(QUBO_to_Ising(dok_qubo.todense().tolist())) # check if this is the same as the input at the top.

    return dok_qubo, embedding
