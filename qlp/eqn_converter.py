"""Converter for linear inequalities to matrices describing the quantum problem
"""
from typing import List, Tuple, Dict

from decimal import Decimal

from numpy import array, ndarray, arange, zeros

from sympy import symbols, Matrix, Symbol
from sympy.matrices import zeros as ZeroMatrix
from sympy.core import relational


def eqns_to_matrix(
    eqns: List[relational.Relational], dependents: List[Symbol]
) -> Tuple[Matrix, Matrix, Matrix]:
    """Converts list of relations to slack variable matrix and vector formalism.

    The result relates to the original eqn such that ``m.deps + v + s = 0``

    For example ``[a1 * x <= b1, a2 * x >= b2]`` becomes

    Code:
        m = | a1 |  and v = | b1 | s = | - s1 |
            | a2 |          | b2 |     | + s2 |

    Arguments:
        eqns:
            List of relational equations. LHS must be linear in dependents, RHS constant.
        dependents:
            List of dependent variables.


    Returns:
        m and v, here m is the Matrix and v the vector containing the slack variable

    """
    dependents = set(dependents)

    mat_coeffs = set()
    vec_coeffs = set()
    for eqn in eqns:
        mat_coeffs = mat_coeffs.union(eqn.lhs.free_symbols.difference(set(dependents)))
        vec_coeffs = vec_coeffs.union(eqn.rhs.free_symbols.difference(set(dependents)))

    n_deps = len(dependents)
    n_eqns = len(eqns)

    s_vars = symbols(f"s(1:{n_eqns+1})")

    v = Matrix([[-eqn.rhs] for eqn in eqns])
    s = Matrix(
        [
            [s if isinstance(eqn, relational.LessThan) else -s]
            for s, eqn in zip(s_vars, eqns)
        ]
    )
    m = ZeroMatrix(rows=n_eqns, cols=n_deps)
    for ne, eqn in enumerate(eqns):
        for nd, dep in enumerate(dependents):
            m[ne, nd] = eqn.lhs.coeff(dep)

    return m, v, s


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


def int_to_bitarray(i: int, bits: int = 8) -> ndarray:
    """Converts an integer to an array where each value corresponds to a bit

    Arguments:
        i: The integer
        bits: The available bits for the substitutin.

    Returns:
        An array of ones and zeros. First element is smallest number
    """
    bit_string = (f"{{0:0{bits}b}}").format(i)
    if len(bit_string) > bits:
        raise ValueError(f"{i} is too large to be represented by {bits} bits")
    return array([int(ii) for ii in bit_string[::-1]])


def get_bit_map(nvars: int, nbits: int) -> ndarray:
    """Creates a map from bit vectors to integers.

    Arguments:
        nvars: Number of vector entries to convert to integers (rows).
        nb: Number of bits for bit vector components (columns = nb * nvar)
    """
    bitmap = 2 ** arange(nbits)
    q = zeros([nvars, nbits * nvars], dtype=int)
    for n in range(nvars):
        q[n, n * nbits : ((n + 1) * nbits)] = bitmap

    return q
