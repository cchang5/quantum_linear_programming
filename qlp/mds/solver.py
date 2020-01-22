"""QUBO solver routines to obtain global optimal solution.
"""
from typing import List, Tuple, Union, Optional
from itertools import product

import numpy as np

from tqdm import tqdm


def classical_search(
    qubo: "SparseMatrix", n_nodes: Optional[int] = None
) -> Union[int, List[Tuple[int]]]:
    """Classicaly searches minimal energy for all integer solutions.

    Assumes that the first n_nodes entries correspond to sorted vertex numbers.

    Arguments:
        qubo: The QUBO to optimize.
        n_nodes: If present, optimizes the spaces for the full qubo but only returns
            solutions accounting for entries from zero to n_nodes (exclusive).
            This means that the energy and slack contributions are excluced from the
            presentation of the solution.

    Returns:
        e_min: The minimal energy.
        solutions: A list of vertices (qubo entries) which are non-zero.
    """
    energies = []
    vecs = []

    n_iterations = 2 ** qubo.shape[1]
    if n_iterations > 10 ** 5:
        go_on = input(
            f"This search will execute {n_iterations} iterations. Continue (y/n)?"
        )
        if go_on.lower() != "y":
            raise KeyboardInterrupt("Too many iterations abort.")

    # Brute force iterate space
    for vv in tqdm(product(*[(0, 1)] * qubo.shape[1])):
        ee = (vv @ qubo @ vv).flatten()[0]
        energies.append(ee)
        vecs.append(vv)

    # Find minimal energy
    e_min = np.min(energies)
    # And corresponding vectors
    solutions = [tuple(vv) for vv, ee in zip(vecs, energies) if ee == e_min]

    if n_nodes is not None:
        # Compute energy for one of the solution vectors by only considering n_nodes
        v0 = np.array([vv if n < n_nodes else 0 for n, vv in enumerate(solutions[0])])
        e_min = (v0 @ qubo @ v0).flatten()[0]
        # Reduce solutions to n_nodes entries
        solutions = [sol[:n_nodes] for sol in solutions]

    # Convert bit list to vertex numbers
    solutions = [
        tuple([n for n, vv in enumerate(sol[:n_nodes]) if vv == 1]) for sol in solutions
    ]

    return e_min, solutions
