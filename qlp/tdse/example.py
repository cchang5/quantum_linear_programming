"""Example scripts for using the tdse module
"""
from typing import List, Dict, Tuple

from numpy import unique, array, concatenate, zeros
from scipy.sparse import dok_matrix
import qlp.mds.transformation as transform


def embed_qubo_example(n_vertices: int) -> Tuple[dok_matrix, Dict[int, List[int]]]:
    """Returns qubo and explicit embedding for NN(n_vertices) graph

    Only implemented for n_vertices=2
    """
    if n_vertices == 2:
        """This is a NN(2) graph embedded in Chimera
        
        Initial Ising form
        h = [0.375  0.375 -0.25 -0.25]
        J = [[0.    0.5   -0.25 -0.25]
             [0.    0.    -0.25 -0.25]
             [0.    0.     0.    0.  ]
             [0.    0.     0.    0.  ]]
        
        Embedded QUBO form from DWave
        640 640 2.5
        641 641 6.0
        643 643 6.0
        645 645 -3.0
        647 647 10.5
        640 645 8.0
        641 645 -4.0
        643 645 -4.0
        640 647 -16.0
        641 647 -4.0
        643 647 -4.0
        
        As matrix
        [[-3    0   8.0 -4.0 -4.0]
         [ 0 10.5 -16.0 -4.0 -4.0]
         [0     0   2.5    0    0]
         [0     0     0  6.0    0]
         [0     0     0    0 6.0]] 
        """
        q = [[-3, 0, 8.0, -4.0, -4.0],
             [0, 10.5, -16.0, -4.0, -4.0],
             [0, 0, 2.5, 0, 0],
             [0, 0, 0, 6.0, 0],
             [0, 0, 0, 0, 6.0]]
        embedding = {0: [645], 1: [647, 640], 2: [641], 3: [643]}
    #    if n_vertices == 2:
    #        """NN(2) on 2000Q_6"""
    #        q = """1890 1890 -0.75
    # 1891 1891 2.625
    # 1892 1892 0.625
    # 1893 1893 1.5
    # 1894 1894 1.5
    # 1890 1892 2.0
    # 1891 1892 -4.0
    # 1890 1893 -1.0
    # 1891 1893 -1.0
    # 1890 1894 -1.0
    # 1891 1894 -1.0"""
    #    embedding = {0: [1892, 1891], 1: [1890], 2: [1893], 3: [1894]}
    else:
        raise ValueError("No embedded graph defined.")

    #nqubits = 0
    #for qubit in embedding:
    #    nqubits += len(embedding[qubit])

    #q = array([[float(i) for i in qn.split(" ")] for qn in q.split("\n")])

    #remap = {
    #    key: idx
    #    for idx, key in enumerate(unique(concatenate((q[:, 0], q[:, 1]), axis=0)))
    #}
    #Qarray = zeros((nqubits, nqubits))
    #for qi in q:
    #    i = remap[qi[0]]
    #    j = remap[qi[1]]
    #    Qarray[i, j] = qi[2]
    dok_qubo = dok_matrix(q)

    # h = dict()
    # J = dict()
    # for qi in q:
    #    i = remap[qi[0]]
    #    j = remap[qi[1]]
    #    if i == j:
    #        h[i] = qi[2]
    #    else:
    #        J[(i,j)] = qi[2]

    # hlist = zeros(nqubits)
    # Jlist = zeros((nqubits, nqubits))
    # for hi in h:
    #    hlist[hi] = h[hi]
    # for Jij in J:
    #    Jlist[Jij[0], Jij[1]] = J[Jij]

    # """NEED TO REPLACE THIS WITH OWN FUNCTION
    # ising_to_qubo ->
    # mds_qlpdb.Ising_to_QUBO(J, h)
    # return dok_matrix
    #
    # from qlp.mds.mds_qlpdb import QUBO_to_Ising
    # Cross check with QUBO_to_Ising(dok_matrix.todense().tolist())
    # to see if we get back same Ising that was put in.
    #
    # """
    # qubo = transform.Ising_to_QUBO(Jlist, hlist)  # currently h and J are dictionaries. you can change them to vector and matrix if you want
    # dok_qubo = dok_matrix(qubo)
    # print("check transform")
    # print(hlist)
    # print(Jlist)
    # print(transform.QUBO_to_Ising(dok_qubo.todense().tolist())) # check if this is the same as the input at the top.

    return dok_qubo, embedding
