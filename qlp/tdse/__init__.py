# pylint: disable=C0103, R0903
"""Library to solve the time dependent schrodinger equation in the many-body Fock space.
"""
import numpy as np
from numpy.linalg import eigh

from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from pandas import read_excel

import matplotlib.pyplot as plt

from qlp.tdse.schedule import AnnealSchedule


class TDSE:
    def __init__(self, n, ising_params, offset_params, solver_params):
        self.n = n
        self.ising = ising_params
        self.offset_params = offset_params
        self.solver_params = solver_params
        self.id2, self.sigx, self.sigz = self.pauli()
        self.FockX, self.FockZ, self.FockZZ = self.init_Fock()
        self.AS = AnnealSchedule(**offset_params)
        self.IsingH = self.constructIsingH(
            self.Bij(self.AS.B(1)) * self.ising["Jij"], self.AS.B(1) * self.ising["hi"]
        )

    def tdse(self, t, y):
        """Define time-dependent Schrodinger equation"""
        f = -1j * np.dot(self.annealingH(t), y)
        return f

    def pauli(self):
        """Pauli matrices"""
        sigx = np.zeros((2, 2))
        sigz = np.zeros((2, 2))
        id2 = np.identity(2)
        sigx[0, 1] = 1.0
        sigx[1, 0] = 1.0
        sigz[0, 0] = 1.0
        sigz[1, 1] = -1.0
        return id2, sigx, sigz

    def ground_state_degeneracy(self, H, degeneracy_tol=1e-6, debug=False):
        eigval, eigv = eigh(H)
        mask = [abs((ei - eigval[0]) / eigval[0]) < degeneracy_tol for ei in eigval]
        gs_idx = np.arange(len(eigval))[mask]
        if debug:
            print(
                f"Num. degenerate states @ s={self.offset_params['normalized_time'][1]}: {len(gs_idx)}"
            )
        return gs_idx, eigval, eigv

    def calculate_overlap(self, psi1, psi2, degen_idx):
        overlap = sum(
            np.absolute([np.dot(np.conj(psi1[:, idx]), psi2) for idx in degen_idx]) ** 2
        )
        return overlap

    def init_eigen(self, type):
        if type == "true":
            # true ground state
            eigvalue, eigvector = eigh(
                self.annealingH(s=self.offset_params["normalized_time"][0])
            )
        elif type == "transverse":
            # DWave initial wave function
            eigvalue, eigvector = eigh(
                -1
                * self.ising["energyscale"]
                * self.constructtransverseH(self.AS.A(0) * np.ones(self.n))
            )
        else:
            raise TypeError("Undefined initial wavefunction.")
        return eigvalue, eigvector

    def init_wavefunction(self, type="transverse"):
        eigvalue, eigvector = self.init_eigen(type)
        y1 = (1.0 + 0.0j) * eigvector[:, 0]
        return y1

    def init_densitymatrix(self, temp=13e-3, type="transverse", debug=False):
        """Initial density matrix
        temperature in kelvins
        """
        kb = 8.617333262145e-5  # Boltzmann constant [eV / K]
        h = 4.135667696e-15  # Plank constant [eV s] (no 2 pi)
        one = 1e-9  # GHz s
        beta = 1 / (temp * kb / h * one)  # inverse temperature [h/GHz]

        # construct initial density matrix
        eigvalue, eigvector = self.init_eigen(type)

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

    def init_Fock(self):
        """Finish all the operators here and store them"""
        FockX = [self.pushtoFock(i, self.sigx) for i in range(self.n)]
        FockZ = [self.pushtoFock(i, self.sigz) for i in range(self.n)]
        FockZZ = [
            [np.dot(FockZ[i], FockZ[j]) for j in range(self.n)] for i in range(self.n)
        ]
        return FockX, FockZ, FockZZ

    def pushtoFock(self, i, local):
        """Push local operator to many-body Fock space"""
        fock = np.identity(1)
        for j in range(self.n):
            if j == i:
                fock = np.kron(fock, local)
            else:
                fock = np.kron(fock, self.id2)
        return fock

    def constructIsingH(self, Jij, hi):
        """Hamiltonian (J_ij is i < j, i.e., upper diagonal"""
        IsingH = np.zeros((2 ** self.n, 2 ** self.n))
        for i in range(self.n):
            IsingH += hi[i] * self.FockZ[i]
            for j in range(i):
                IsingH += Jij[j, i] * self.FockZZ[i][j]
        return IsingH

    def constructtransverseH(self, hxi):
        transverseH = np.zeros((2 ** self.n, 2 ** self.n))
        for i in range(self.n):
            transverseH += hxi[i] * self.FockX[i]
        return transverseH

    def Bij(self, B):
        """Annealing Hamiltonian
        https://www.dwavesys.com/sites/default/files/14-1002A_B_tr_Boosting_integer_factorization_via_quantum_annealing_offsets.pdf
        Equation (1)
        """
        return np.asarray(
            [[np.sqrt(B[i] * B[j]) for i in range(self.n)] for j in range(self.n)]
        )

    def annealingH(self, s):
        AxtransverseH = self.constructtransverseH(self.AS.A(s) * np.ones(self.n))
        BxIsingH = self.constructIsingH(
            self.Bij(self.AS.B(s)) * self.ising["Jij"], self.AS.B(s) * self.ising["hi"]
        )
        H = self.ising["energyscale"] * (-1 * AxtransverseH + BxIsingH)
        return H

    def solve_pure(self, y1, ngrid=11, debug=False):
        start = self.offset_params["normalized_time"][0]
        end = self.offset_params["normalized_time"][1]
        interval = np.linspace(start, end, ngrid)

        sol = pure_sol_interface(y1)

        for jj in range(ngrid - 1):
            y1 = y1 / (np.sqrt(np.absolute(np.dot(np.conj(y1), y1))))
            tempsol = solve_ivp(
                self.tdse, [interval[jj], interval[jj + 1]], y1, **self.solver_params
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

    def annealingH_densitymatrix(self, s):
        Fockid = np.identity(self.Focksize)
        return np.kron(self.annealingH(s), Fockid) - np.kron(Fockid, self.annealingH(s))

    def f_densitymatrix(self, t, y):
        """Define time-dependent Schrodinger equation for density matrix"""
        f = -1j * np.dot(self.annealingH_densitymatrix(t), y)
        return f

    def f_densitymatrix2(self, t, y):
        """Define time-dependent Schrodinger equation for density matrix"""
        # f = -1j * np.dot(self.annealingH_densitymatrix(t), y)
        # print('waht', type(self.Focksize))
        # print(self.Focksize)
        ymat = y.reshape((self.Focksize, self.Focksize))
        H = self.annealingH(t)
        ymat = -1j * (np.dot(H, ymat) - np.dot(ymat, H))
        f = ymat.reshape(self.Focksize ** 2)
        return f

    def solve_mixed(self, rho):
        self.Focksize = int(np.sqrt(len(rho)))
        sol = solve_ivp(
            self.f_densitymatrix2, self.offset_params["normalized_time"], rho
        )
        return sol


class pure_sol_interface:
    def __init__(self, y1):
        self.t = np.zeros((0))
        self.y = np.zeros((y1.size, 0))


def embed_qubo_example(nvertices):
    if nvertices == 2:
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
    else:
        raise ValueError("No embedded graph defined.")
    q = np.array([[float(i) for i in qn.split(" ")] for qn in q.split("\n")])
    from scipy.sparse import dok_matrix

    remap = {
        key: idx
        for idx, key in enumerate(np.unique(np.concatenate((q[:, 0], q[:, 1]), axis=0)))
    }
    qubo = dok_matrix((len(np.unique(q[:, 0])), len(np.unique(q[:, 0]))), dtype=float)
    for qi in q:
        i = remap[qi[0]]
        j = remap[qi[1]]
        qubo[i, j] = qi[2]
    return qubo, embedding
