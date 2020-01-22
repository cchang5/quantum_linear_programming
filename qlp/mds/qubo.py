"""Tools to generate a Minimal Dominating Set QUBO from a graph
"""
from typing import Set, Tuple, List
from scipy.sparse import dok_matrix, bmat
import numpy as np


def get_adjacency(graph: Set[Tuple[int]]) -> dok_matrix:
    """This routine computes the adjecency matrix for a given graph.

    It assumes the graph is connected and nodes are labeled from 0...N-1.
    Self loops are forbidden.
    """
    if not isinstance(graph, set):
        raise TypeError("Graph must be a set of tuples")

    nodes = set()
    adj = dict()

    for v1, v2 in graph:
        if v1 == v2:
            raise KeyError(f"Self loops are not allowed (v={v1}).")

        nodes.add(v1)
        nodes.add(v2)
        adj[(v1, v2)] = 1

    nodes = sorted(nodes)
    n_nodes = len(nodes)

    if nodes != list(range(n_nodes)):
        raise KeyError("Nodes must be from 0 ... n_nodes")

    adjacency = dok_matrix((n_nodes, n_nodes), dtype=int)
    for (v1, v2), val in adj.items():
        adjacency[(v1, v2)] = val
        adjacency[(v2, v1)] = val

    if not all([val == 1 for val in adjacency.values()]):
        raise KeyError("Double entries in graph")

    return adjacency


def get_bitmap(n_neigbors: List[int]) -> dok_matrix:
    r"""Computes the map of slack-bits to slack integers.

    This matrix B maps the bit vector b to slack variables s.
    Each s_i has max(s_i) >= N_i where N_i is the number of
    neigbors of vertex i.
    The bit vector b = (b_00, b_01, b_02, ... , b_11, b_12, ...)
    with b_ij will be mapped onto s_iaccording to
    s_i = sum_j 2**j b_ij.
    Thus the size of the returned matrix scales as
    ~ sum_i log2(N_i)

    Arguments:
        n_neigbors: The number of neighbors for each vertex.
    """
    n_nodes = len(n_neigbors)

    # Figure out how many bits we need for each neighborhood
    n_bits = np.floor(np.log2(n_neigbors) + 1).astype(int)

    print(n_bits)

    bitmap = dok_matrix((n_nodes, n_bits.sum()), dtype=int)

    acc = 0
    for v, n_bit in enumerate(n_bits):
        for n in range(n_bit):
            bitmap[(v, acc)] = 2 ** n
            acc += 1

    return bitmap


def get_mds_qubo(graph: Set[Tuple[int]]) -> dok_matrix:
    """This routine computes minimal dominating set QUBO for a given graph.

    It assumes the graph is connected and nodes are labeled from 0...N-1.
    Self loops are forbidden.
    """
    ## This is N
    adjacency = get_adjacency(graph)

    ## Id in x-space
    one = dok_matrix(adjacency.shape, dtype=int)
    for n in range(adjacency.shape[0]):
        one[(n, n)] = 1

    ## |N|
    n_neigbors = adjacency.sum(axis=1).flatten().tolist()[0]

    ## diag(|N|)
    diag_neighbor = dok_matrix(adjacency.shape, dtype=int)
    for n, el in enumerate(n_neigbors):
        diag_neighbor[(n, n)] = el

    ## B
    bitmap = get_bitmap(n_neigbors)
    n_bits = bitmap.shape[1]

    ## diag(|B|)
    diag_bitmap = dok_matrix((n_bits, n_bits), dtype=int)
    for n, el in enumerate(bitmap.sum(axis=0).tolist()[0]):
        diag_bitmap[(n, n)] = el

    ## Compute QUBO components
    alpha = -one + 2 * adjacency - 2 * diag_neighbor + adjacency @ adjacency
    beta = -2 * (one + adjacency) @ bitmap
    gamma = bitmap.T @ bitmap + 2 * diag_bitmap

    ## Multiply by penalty factor
    penalty = len(n_neigbors) + 1
    alpha *= penalty
    beta *= penalty
    gamma *= penalty

    ## Add in minimization condition
    alpha += one

    ## Construt QUBO
    return bmat([[alpha, beta], [None, gamma]]).todok()


def main():  # pylint: disable=R0914
    """Generates a random graph
    """
    import matplotlib.pylab as plt
    from qlp.mds.graph_tools import generate_graph, get_plot_mpl
    from qlp.mds.solver import classical_search

    n_nodes = 5
    n_edges = 5
    n_edge_max = 3

    test_graph = generate_graph(n_nodes, n_edges, n_edge_max=n_edge_max)
    qubo = get_mds_qubo(test_graph)
    e_min, solutions = classical_search(qubo, n_nodes=n_nodes)

    # Plotting
    n_sols = len(solutions)
    col_wrap = 4
    n_rows = max(n_sols // col_wrap, 1)
    n_cols = col_wrap if col_wrap > n_sols else n_sols

    fig, axs = plt.subplots(ncols=n_cols, nrows=n_rows)
    for sol, ax in zip(solutions, axs.flatten()):
        get_plot_mpl(test_graph, color_nodes=sol, ax=ax)

    fig.suptitle(rf"$\gamma(G) \leq {e_min}$")
    plt.show()


if __name__ == "__main__":
    main()
