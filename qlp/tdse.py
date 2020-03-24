# Library to solve the time dependent schrodinger equation in the many-body Fock space.
from pandas import read_excel
import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from qlp.mds.mds_qlpdb import QUBO_to_Ising, find_offset, AnnealOffset

# Get DWave anneal schedule
class s_to_offset:
    def __init__(self, fill_value):
        if fill_value == "extrapolate":
            fill_valueC = fill_value
            fill_valueA = fill_value
            fill_valueB = fill_value
        elif fill_value == "truncate":
            fill_valueC = (0, 1)
            fill_valueA = (10.3214800, 0.0000026)
            fill_valueB = (0.4927432, 11.8604700)
        paramsC = {
            "kind": "linear",
            "fill_value": fill_valueC,
        }  # linear makes for more sensible extrapolation.
        paramsA = {
            "kind": "linear",
            "fill_value": fill_valueA,
        }  # linear makes for more sensible extrapolation.
        paramsB = {
            "kind": "linear",
            "fill_value": fill_valueB,
        }  # linear makes for more sensible extrapolation.
        self.anneal_schedule = read_excel(
            io="./09-1212A-B_DW_2000Q_5_anneal_schedule.xlsx", sheet_name=1
        )
        self.interpC = interp1d(self.anneal_schedule["s"], self.anneal_schedule["C (normalized)"], **paramsC)
        self.interpA = interp1d(self.anneal_schedule["C (normalized)"], self.anneal_schedule["A(s) (GHz)"], **paramsA)
        self.interpB = interp1d(self.anneal_schedule["C (normalized)"], self.anneal_schedule["B(s) (GHz)"], **paramsB)

    def sanity_check(self):
        # Sanity check: The data and interpolation should match
        # Interpolate C
        plt.figure()
        ax = plt.axes()
        x = np.linspace(0, 1)
        ax.errorbar(
            x=self.anneal_schedule["s"], y=self.anneal_schedule["C (normalized)"]
        )
        ax.errorbar(x=x, y=self.interpC(x))
        plt.draw()
        plt.show()
        # Interpolate A and B
        plt.figure()
        ax = plt.axes()
        x = np.linspace(-0.05, 1.05)
        ax.errorbar(
            x=self.anneal_schedule["s"],
            y=self.anneal_schedule["A(s) (GHz)"]
            / self.anneal_schedule["A(s) (GHz)"].max(),
        )
        ax.errorbar(x=x, y=self.interpA(self.interpC(x)))
        ax.errorbar(
            x=self.anneal_schedule["s"],
            y=self.anneal_schedule["B(s) (GHz)"]
            / self.anneal_schedule["B(s) (GHz)"].max(),
        )
        ax.errorbar(x=x, y=self.interpB(self.interpC(x)))
        plt.draw()
        plt.show()


class AnnealSchedule:
    def __init__(self, offset, hi, offset_min, offset_range, fill_value="extrapolate"):
        AO = AnnealOffset(offset)
        self.offset_list, self.offset_tag = AO.fcn(hi, offset_min, offset_range)
        self.s2o = s_to_offset(fill_value)

    def C(self, s):
        C = self.s2o.interpC(s)
        C_offset = C + self.offset_list
        return C_offset

    def A(self, s):
        C = self.C(s)
        return self.s2o.interpA(C)

    def B(self, s):
        C = self.C(s)
        return self.s2o.interpB(C)


class TDSE:
    def __init__(self, n, ising_params, offset_params):
        self.n = n
        self.ising = ising_params
        self.id2, self.sigx, self.sigz = self.pauli()
        self.FockX, self.FockZ, self.FockZZ = self.init_Fock()
        self.AS = AnnealSchedule(hi=ising_params["hi"], **offset_params)
        self.IsingH = self.constructIsingH(
            self.Bij(self.AS.B(1)) * self.ising["Jij"], self.AS.B(1) * self.ising["hi"]
        )

    def __call__(self, t, y):
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
        """Annealing Hamiltonian"""
        return np.asarray(
            [[0.5 * (B[i] + B[j]) for i in range(self.n)] for j in range(self.n)]
        )

    def annealingH(self, s):
        AxtransverseH = self.constructtransverseH(self.AS.A(s) * np.ones(self.n))
        BxIsingH = self.constructIsingH(
            self.Bij(self.AS.B(s)) * self.ising["Jij"], self.AS.B(s) * self.ising["hi"]
        )
        H = self.ising["energyscale"] * (-0.5 * AxtransverseH + 0.5 * BxIsingH)
        return H


def embed_qubo_example():
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
    return qubo
