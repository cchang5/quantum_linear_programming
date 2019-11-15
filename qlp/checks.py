"""Cross checks for integer map problems
"""
from typing import List

from itertools import product

from sympy import Matrix, Symbol
from sympy.core.relational import Relational

from pandas import DataFrame, Series

import qlp.eqn_converter as qec


def generate_constrained_table(
    inequalities: List[Relational], dependents: List[Symbol], n_bits: int
) -> DataFrame:
    """Evaluates the constrained for all allowed bit vectors in the matrix basis

    Arguments:
        inequalities: All inequalities constraining the problem.
        dependents: The variable expressions within the constraints.
        n_bits: The number of bits to represent the variables.

    Returns:
        DataFrame with a column for each variabe the constrained value as
        ``psi.T @ constrained @ psi`` (missing a constant shift) and a column with this
        value shifted.
    """
    xi, _ = qec.get_basis(dependents, n_constraints=len(inequalities))
    alpha, beta = qec.constraints_to_matrix(inequalities, dependents)
    q = qec.get_bit_map(len(xi), n_bits)
    constrained_mat = qec.get_constrained_matrix(q, alpha, beta)

    data = []
    for psi_vec in product(*[range(2)] * q.shape[1]):
        psi_vec = Matrix(psi_vec)
        xi_vec = q @ psi_vec
        constrained = psi_vec.T @ constrained_mat @ psi_vec
        info = {expr: val for expr, val in zip(xi, xi_vec)}
        info["constrained"] = constrained[0]
        info["shifted_constrained"] = (constrained + beta.T @ beta)[0]
        data.append(info)

    return DataFrame(data).sort_values("shifted_constrained", ascending=True)


def check_inequalities(
    data: Tuple[Dict[Any, Any], Series], inequalities: List[Relational]
) -> bool:
    """Checks wether all inequalities are fulfilled by dependents.

    Arguments:
        data:
            Dict or series containing key value pair for all dependents in all
            inequalities. Must contain only those keys or keys which contain
            `constrained`.
        inequalities:
            List of all inequalities.
    """
    if isinstance(data, Series):
        data = data.to_dict()
    dependents = [var for var in data if "constrained" not in str(var)]
    return bool(
        all(
            [ineq.subs({dep: data[dep] for dep in dependents}) for ineq in inequalities]
        )
    )


def run_checks(  # pylint: disable=C0103
    df: DataFrame, inequalities: List[Relational]
) -> bool:
    """Runs the following checks on a constraints table

    1. Checks if all shifted constrained are larger or equal to zero.
    2. If all inequalities are true for shifted constrained == 0 parameters.
    3. If any inequality is false for shifted constrained > 0 parameters.

    Raise:
        AssertionError error if not all checks are fulfilled
    """
    if not all(df.shifted_constrained.unique() >= 0):
        raise AssertionError(
            "Not all values of shifted constrained are larger or equal to zero."
        )

    if not (
        df.query("shifted_constrained == 0")
        .apply(lambda row: check_inequalities(row, inequalities), axis=1,)
        .astype(bool)
        .all()
    ):
        raise AssertionError(
            "Not all values where the shifted constraints are zero"
            " fulfill all inequalities."
        )

    if (
        df.query("shifted_constrained > 0")
        .apply(lambda row: check_inequalities(row, inequalities), axis=1,)
        .astype(bool)
        .all()
    ):
        raise AssertionError(
            "Not all values where the shifted constraints are larger than zero"
            " do not fulfill all inequalities."
        )

    return True
