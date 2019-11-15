"""Helper function for plotting
"""
from typing import List

import numpy as np
from sympy import Matrix, S
from sympy.core.relational import Relational

import qlp.eqn_converter as qe

DEPENDENTS = [S("x_0"), S("x_1")]


def f(x1: float, x2: float) -> float:
    r"""Hyperbole function

    Implements
    $$
        f(x_1, x_2) = 8 -(x_1 - 2)^2 - (x_2 - 2)^2
    $$
    """
    return 8 - (x1 - 2) ** 2 - (x2 - 2) ** 2


def get_omega_0(n_bits: int, as_numeric: bool = False) -> Matrix:
    r"""Computes the matrix which represents $f$ as a bit matrix-vector operation

    E.g.,
    $$
        f(x_1, x_2) = \psi_\alpha M_{\alpha \beta} \psi_\beta
        \, , \qquad
        M_{\alpha \beta}
        =
        - Q_{1\alpha}Q_{1\beta}
        - Q_{2\alpha}Q_{2\beta}
        + 4Q_{1\alpha}\delta_{\alpha\beta}
        + 4Q_{2\alpha}\delta_{\alpha\beta}
    $$
    where $\psi$ is a bit vector.

    Arguments:
        n_bits: Number of bits
        as_numeric: Return numpy array if True. Else sympy expression.
    """
    q = qe.get_bit_map(len(DEPENDENTS), n_bits)
    mat = -q.T @ q + 4 * np.diag(np.sum(q, axis=0))
    return np.array(mat) if as_numeric else mat


def get_omega(
    inequalities: List[Relational], n_bits: int, p: int = 10, as_numeric: bool = False
) -> Matrix:
    """
    """
    n_vars = len(DEPENDENTS) + len(inequalities)
    q = qe.get_bit_map(n_vars=n_vars, n_bits=n_bits)
    a, b = qe.constraints_to_matrix(
        inequalities, dependents=DEPENDENTS, as_numeric=as_numeric
    )
    omega = -p * qe.get_constrained_matrix(q, a, b, as_numeric=as_numeric)
    nx = len(DEPENDENTS) * n_bits
    omega[:nx, :nx] += get_omega_0(n_bits, as_numeric=as_numeric)
    return omega
