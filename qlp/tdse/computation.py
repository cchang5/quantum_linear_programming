# pylint: disable=C0103, R0902, R0903
"""Library to solve the time dependent schrodinger equation in the many-body Fock space.

This module contains the core computations
"""
from typing import Dict, Any, Tuple, List

from numpy import ndarray
import numpy as np
from numpy.linalg import eigh

from scipy.integrate import solve_ivp

from qlp.tdse.schedule import AnnealSchedule


def _set_up_pauli():
    """Creates Pauli matrices and identity
    """
    sigx = np.zeros((2, 2))
    sigz = np.zeros((2, 2))
    id2 = np.identity(2)
    sigx[0, 1] = 1.0
    sigx[1, 0] = 1.0
    sigz[0, 0] = 1.0
    sigz[1, 1] = -1.0
    return id2, sigx, sigz


ID2, SIG_X, SIG_Z = _set_up_pauli()


class PureSolutionInterface:
    """Interface for a pure state solution

    Attributes:
        t: array
        y: array
    """

    def __init__(self, y1):
        self.t = np.zeros((0))
        self.y = np.zeros((y1.size, 0))


class TDSE:
    """Time dependent Schrödinger equation solver class

    .. code-block:: python

        tdse = TDSE(n, ising_params, offset_params, solver_params)

        # Get thermodynamical state density
        temperature=15e-3
        initial_wavefunction = "true"
        rho = tdse.init_densitymatrix(temperature, initial_wavefunction)

        # Compute anneal
        sol_densitymatrix = tdse.solve_mixed(rho)
    """

    def __init__(
        self,
        n: int,
        ising_params: Dict[str, Any],
        offset_params: Dict[str, Any],
        solver_params: Dict[str, Any],
    ):
        """Init the class with

        Arguments:
            n: Number of qubits
            ising_params: Parameters for the ising model, e.g., keys are
                {"Jij", "hi", "c", "energyscale"}.
            offset_params: Parameters for AnnealSchedule
            solver_params: Parameters for solve_ivp
        """
        self.n = n
        self.ising = ising_params
        self.offset_params = offset_params
        self.solver_params = solver_params
        self.FockX, self.FockZ, self.FockZZ = self._init_Fock()
        self.Focksize = None
        self.AS = AnnealSchedule(**offset_params)
        self.IsingH = self._constructIsingH(
            self._Bij(self.AS.B(1)) * self.ising["Jij"], self.AS.B(1) * self.ising["hi"]
        )

    def _apply_H(self, t, psi: ndarray) -> ndarray:
        """Computes `i H(t) psi`"""
        return -1j * np.dot(self.annealingH(t), psi)

    def ground_state_degeneracy(
        self, H: ndarray, degeneracy_tol: float = 1e-6, debug: bool = False
    ) -> Tuple[ndarray, ndarray, ndarray]:
        """Computes the number of degenerate ground states

        Identifies degeneracy by comparing closeness to smallest eigenvalule.

        Arguments:
            H: Hamiltonian to compute eigen vectors off
            degeneracy_tol: Precision of comparison to GS
            debug: More output

        Returns: Ids for gs vectors, all eigenvalues, all eigenvectors
        """
        eigval, eigv = eigh(H)
        mask = [abs((ei - eigval[0]) / eigval[0]) < degeneracy_tol for ei in eigval]
        gs_idx = np.arange(len(eigval))[mask]
        if debug:
            print(
                f"Num. degenerate states @"
                f" s={self.offset_params['normalized_time'][1]}: {len(gs_idx)}"
            )
        return gs_idx, eigval, eigv

    @staticmethod
    def calculate_overlap(psi1: ndarray, psi2: ndarray, degen_idx: List[int]) -> float:
        """Computes overlaps of states in psi1 with psi2 (can be multiple)

        Overlap is defined as ``sum(<psi1_i | psi2>, i in degen_idx)``. This allows to
        compute overlap in presence of degeneracy. See als eq. (57) in the notes.

        Arguments:
            psi1: Set of wave vectors. Shape is (size_vector, n_vectors).
            psi2: Vector to compare against.
            degen_idx: Index for psi1 vectors.

        """
        return sum(
            np.absolute([np.dot(np.conj(psi1[:, idx]), psi2) for idx in degen_idx]) ** 2
        )

    def init_eigen(self, dtype: str) -> Tuple[ndarray, ndarray]:
        """Computes eigenvalue and vector of initial Hamiltonian either as a pure
        eigenstate of `H_init` (transverse) or a s a superposition of
        `A(s) H_init + B(s) H_final` (true).
        """
        if dtype == "true":
            # true ground state
            eigvalue, eigvector = eigh(
                self.annealingH(s=self.offset_params["normalized_time"][0])
            )
        elif dtype == "transverse":
            # DWave initial wave function
            eigvalue, eigvector = eigh(
                -1
                * self.ising["energyscale"]
                * self._constructtransverseH(self.AS.A(0) * np.ones(self.n))
            )
        else:
            raise TypeError("Undefined initial wavefunction.")
        return eigvalue, eigvector

    def init_wavefunction(self, dtype="transverse") -> ndarray:
        """Returns wave function for first eigenstate of Hamiltonian of dtype.
        """
        _, eigvector = self.init_eigen(dtype)
        return (1.0 + 0.0j) * eigvector[:, 0]

    def init_densitymatrix(
        self, temp: float = 13e-3, dtype: str = "transverse", debug: bool = False
    ) -> ndarray:
        """Returns density matrix for s=0

        ``rho(s=0) = exp(- beta H(0)) / Tr(exp(- beta H(0)))``


        Arguments:
            temp: Temperature in K
            dtype: Kind of inital wave function (true or transverse)
            debug: More output messages
        """
        kb = 8.617333262145e-5  # Boltzmann constant [eV / K]
        h = 4.135667696e-15  # Plank constant [eV s] (no 2 pi)
        one = 1e-9  # GHz s
        beta = 1 / (temp * kb / h * one)  # inverse temperature [h/GHz]

        # construct initial density matrix
        eigvalue, eigvector = self.init_eigen(dtype)

        dE = eigvalue[:] - eigvalue[0]
        pr = np.exp(-beta * dE)
        pr = pr / sum(pr)
        if debug:
            print("dE", dE)
            print("pr", pr, "total", sum(pr))

        rho = np.zeros((eigvalue.size * eigvalue.size))
        for i in range(eigvalue.size):
            rho = rho + (pr[i]) * np.kron(eigvector[:, i], np.conj(eigvector[:, i]))
        rho = (1.0 + 0.0j) * rho
        return rho

    def _init_Fock(self) -> Tuple[ndarray, ndarray, ndarray]:
        r"""Computes pauli matrix tensor products

        Returns:
            ``sigma^x_i \otimes 1``,
            ``sigma^z_i \otimes 1``,
            ``sigma^z_i \otimes sigma^z_j \otimes 1``
        """
        FockX = [self.pushtoFock(i, SIG_X) for i in range(self.n)]
        FockZ = [self.pushtoFock(i, SIG_Z) for i in range(self.n)]
        FockZZ = [
            [np.dot(FockZ[i], FockZ[j]) for j in range(self.n)] for i in range(self.n)
        ]
        return FockX, FockZ, FockZZ

    def pushtoFock(self, i: int, local: ndarray) -> ndarray:
        """Tensor product of `local` at particle index i with 1 in fock space

        Arguments:
            i: particle index of matrix local
            local: matrix operator
        """
        fock = np.identity(1)
        for j in range(self.n):
            if j == i:
                fock = np.kron(fock, local)
            else:
                fock = np.kron(fock, ID2)
        return fock

    def _constructIsingH(self, Jij: ndarray, hi: ndarray) -> ndarray:
        """Computes Hamiltonian (``J_ij is i < j``, i.e., upper diagonal"""
        IsingH = np.zeros((2 ** self.n, 2 ** self.n))
        for i in range(self.n):
            IsingH += hi[i] * self.FockZ[i]
            for j in range(i):
                IsingH += Jij[j, i] * self.FockZZ[i][j]
        return IsingH

    def _constructtransverseH(self, hxi: ndarray) -> ndarray:
        r"""Construct sum of tensor products of ``\sigma^x_i \otimes 1``
        """
        transverseH = np.zeros((2 ** self.n, 2 ** self.n))
        for i in range(self.n):
            transverseH += hxi[i] * self.FockX[i]
        return transverseH

    def _Bij(self, B: ndarray) -> ndarray:
        """J_ij coefficients for final Annealing Hamiltonian

        https://www.dwavesys.com/sites/default/files/
          14-1002A_B_tr_Boosting_integer_factorization_via_quantum_annealing_offsets.pdf

        Equation (1)

        Arguments:
            B: Anneal coefficients for given schedule.
        """
        return np.asarray(
            [[np.sqrt(B[i] * B[j]) for i in range(self.n)] for j in range(self.n)]
        )

    def annealingH(self, s: float) -> ndarray:
        """Computes ``H(s) = A(s) H_init + B(s) H_final`` in units of "energyscale"
        """
        AxtransverseH = self._constructtransverseH(self.AS.A(s) * np.ones(self.n))
        BxIsingH = self._constructIsingH(
            self._Bij(self.AS.B(s)) * self.ising["Jij"], self.AS.B(s) * self.ising["hi"]
        )
        H = self.ising["energyscale"] * (-1 * AxtransverseH + BxIsingH)
        return H

    def solve_pure(
        self, y1: ndarray, ngrid: int = 11, debug: bool = False
    ) -> PureSolutionInterface:
        """Solves time depepdent Schrödinger equation for pure inital state
        """
        start = self.offset_params["normalized_time"][0]
        end = self.offset_params["normalized_time"][1]
        interval = np.linspace(start, end, ngrid)

        sol = PureSolutionInterface(y1)

        for jj in range(ngrid - 1):
            y1 = y1 / (np.sqrt(np.absolute(np.dot(np.conj(y1), y1))))
            tempsol = solve_ivp(
                self._apply_H,
                [interval[jj], interval[jj + 1]],
                y1,
                **self.solver_params,
            )
            y1 = tempsol.y[:, tempsol.t.size - 1]
            sol.t = np.hstack((sol.t, tempsol.t))
            sol.y = np.hstack((sol.y, tempsol.y))

        if debug:
            print(
                "final total prob",
                (np.absolute(np.dot(np.conj(sol.y[:, -1]), sol.y[:, -1]))) ** 2,
            )
        return sol

    def _annealingH_densitymatrix(self, s: float) -> ndarray:
        """Tensor product of commutator of annealing Hamiltonian with id in Fock space

        Code:
            H(s) otimes 1 - 1 otimes H(s)
        """
        Fockid = np.identity(self.Focksize)
        return np.kron(self.annealingH(s), Fockid) - np.kron(Fockid, self.annealingH(s))

    def _apply_tdse_dense(self, t: float, y: ndarray) -> ndarray:
        """Computes ``-i [H(s), rho(s)]`` for density vector `y`"""
        f = -1j * np.dot(self._annealingH_densitymatrix(t), y)
        return f

    def _apply_tdse_dense2(self, t: float, y: ndarray) -> ndarray:
        """Computes ``-i [H(s), rho(s)]`` for density vector ``y`` by reshaping ``y``
        """
        # f = -1j * np.dot(self.annealingH_densitymatrix(t), y)
        # print('waht', type(self.Focksize))
        # print(self.Focksize)
        ymat = y.reshape((self.Focksize, self.Focksize))
        H = self.annealingH(t)
        ymat = -1j * (np.dot(H, ymat) - np.dot(ymat, H))
        f = ymat.reshape(self.Focksize ** 2)
        return f

    def solve_mixed(self, rho: ndarray) -> ndarray:
        """Solves the TDSE
        """
        self.Focksize = int(np.sqrt(len(rho)))
        return solve_ivp(
            self._apply_tdse_dense2, self.offset_params["normalized_time"], rho
        )

    # Compute Correlations
    # One time correlation function
    def cZ(self, ti, xi, sol_densitymatrix):
        return np.trace(
            np.dot(
                self.FockZ[xi],
                sol_densitymatrix.y[:, ti].reshape(2 ** self.n, 2 ** self.n),
            )
        )

    def cZZ(self, ti, xi, xj, sol_densitymatrix):
        return np.trace(
            np.dot(
                self.FockZZ[xi][xj],
                sol_densitymatrix.y[:, ti].reshape(2 ** self.n, 2 ** self.n),
            )
        )

    def cZZd(self, ti, xi, xj, sol_densitymatrix):
        return self.cZZ(ti, xi, xj, sol_densitymatrix) - self.cZ(
            ti, xi, sol_densitymatrix
        ) * self.cZ(ti, xj, sol_densitymatrix)

    # Two time correlation function
    # http://qutip.org/docs/latest/guide/guide-correlation.html
    def cZZt2(self, ti, xi, sol_densitymatrixt2):
        return np.trace(
            np.dot(
                self.FockZ[xi],
                sol_densitymatrixt2.y[:, ti].reshape(2 ** self.n, 2 ** self.n),
            )
        )

    def cZZt2d(self, ti, xi, tj, xj, sol_densitymatrix, sol_densitymatrixt2):
        return np.trace(
            np.dot(
                self.FockZ[xi],
                sol_densitymatrixt2.y[:, ti].reshape(2 ** self.n, 2 ** self.n),
            )
        ) - self.cZ(ti, xi, sol_densitymatrix) * self.cZ(tj, xj, sol_densitymatrix)
