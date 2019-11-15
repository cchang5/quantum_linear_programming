"""Converter for linear inequalities to matrices describing the quantum problem
"""
from typing import List, Tuple, Dict, Optional

from decimal import Decimal

import numpy as np

from sympy import Matrix, Symbol, S
from sympy.matrices import zeros as ZeroMatrix
from sympy.matrices import diag
from sympy.core import relational


def get_basis(
    dependents: List[Symbol], n_constraints: int
) -> Tuple[Matrix, np.ndarray]:
    """Converts dependent variables and number of constraints to a joined problem vector.

    Arguments:
        dependents: List of dependent variables. Will conserve the order.
        n_constraints: Number of inequalities constraining the problem.

    Returns:
        xi:
            The joined vector of dependent and slack variables
        xi_to_x:
            A matrix which converts xi to an x vector.
            The vector has the same size as xi but the slack variable components are
            zero.

    Example:
        ```
        xi, xi_to_x = get_integer_basis(dependents=[x0, x1], n_constraints=1)

        xi == [x0, x1, s0]
        xi_to_x@xi == [x0, x1, 0]
        ```
    """
    xi = Matrix(
        [dep for dep in dependents] + [S(f"s_{i}") for i in range(n_constraints)]
    )
    xi_to_x = np.diag(
        ([1 for i in range(len(dependents))] + [0 for i in range(n_constraints)])
    )

    return xi, xi_to_x


def constraints_to_matrix(
    inequalities: List[relational.GreaterThan], dependents: List[Symbol]
) -> Tuple[Matrix, Matrix]:
    """Converts list of linear inequalities to slack variable matrix-vector equations.

    The result relates to the original eqn such that ``m@(deps, s) + b = 0``

    Arguments:
        inequalities:
            List of relational equations. LHS must be linear in dependents, RHS constant.
        dependents:
            List of dependent variables.

    Returns:
        m and v, here m is the Matrix and v the vector containing the slack variable

    Example:
        For example ``[a1 * x - b1 <= 0, a2 * x - b2 >= 0)]`` becomes

        ```
        a = | +a1  s1  0  |  and b = | +b1 |
            | -a2  0   s2 |          | -b2 |
        ```
    """
    for ieqn in inequalities:
        if not ieqn.rhs == 0:
            raise TypeError("RHS of inequality must be zero. Received %s" % ieqn.rhs)

    n_deps = len(dependents)
    n_eqns = len(inequalities)

    a = ZeroMatrix(rows=n_eqns, cols=(n_deps + n_eqns))
    b = ZeroMatrix(rows=n_eqns, cols=1)

    for ne, ieqn in enumerate(inequalities):

        if isinstance(ieqn, relational.LessThan):
            ieqn = ieqn.lhs * (-1) >= 0
        elif isinstance(ieqn, relational.GreaterThan):
            pass
        else:
            raise TypeError(
                "All inequalities must be either of form '>=' or '<='. Received %s."
                % type(ieqn)
            )

        for nd, dep in enumerate(dependents):
            a[ne, nd] = ieqn.lhs.coeff(dep)
        a[ne, n_deps + ne] = -1
        b[ne, 0] = ieqn.lhs.subs({dep: 0 for dep in dependents})

    return a, b


def get_bit_map(nvars: int, nbits: int) -> np.ndarray:
    """Creates a map from bit vectors to integers.

    Arguments:
        nvars: Number of vector entries to convert to integers (rows).
        nb: Number of bits for bit vector components (columns = nb * nvar)
    """
    bitmap = 2 ** np.arange(nbits)
    q = np.zeros([nvars, nbits * nvars], dtype=int)
    for n in range(nvars):
        q[n, n * nbits : ((n + 1) * nbits)] = bitmap

    return q


def get_bit_vector(n_vars: int, n_bits: int) -> Matrix:
    """Constructs a vector of bits ``psi_ij`` for input space.

    Vector is of size ``n_vars x n_bits``.

    Arguments:
        n_vars: Number of variables (first index).
        n_bits: Number of bits. The maximal value of the variable is ``2**n_bits - 1``.
    """
    return Matrix([f"psi_{i}{j}" for i in range(n_vars) for j in range(n_bits)])


def get_constrained_matrix(  # pylint: disable=C0103
    q: Matrix, a: Matrix, b: Optional[Matrix] = None, as_numeric: bool = False
) -> Tuple[Matrix, np.ndarray]:
    """Computes the linear constrained in a bit basis.

    The constrained term is assumed to be of the form
    ```
    constraint = (a @ xi + b).T @ (a @ xi + b)
    ```

    Arguments:
        q: The bit to constraint vector map.
        a: The linear component of the constraint.
        b: The constant term of the component.
        as_numeric: Convert sympy matrix to float if possible.

    Returns:
        Matrix ``m`` such that ``psi.T @ m @ psi + b.T @ b = constraint`` where
        ``xi = q @ psi``.
    """
    mat = q.T @ a.T @ a @ q
    if b is not None:
        mat += diag(*b.T @ a @ q) + diag(*q.T @ a.T @ b)
    return np.array(mat) if as_numeric else mat


def rescale_expressions(expr: Symbol, subs: Dict[str, str]) -> Symbol:
    """Rescales and substitutes all values.

    The values are multiplied by 10**power such that all values are integers.

    Arguments:
        expr: The expression to substitute
        subs: The symbol to value map. Must be strings.

    Returns:
        The rescaled and substituded expression
    """
    max_neg_power = 0
    for val in subs.values():
        exponent = Decimal(val).as_tuple().exponent
        max_neg_power = exponent if exponent < max_neg_power else max_neg_power

    fact = 10 ** (-max_neg_power)

    print(f"Multipying by {fact}")

    rescaled_subs = {par: int(Decimal(val) * fact) for par, val in subs.items()}

    return expr.subs(rescaled_subs)
