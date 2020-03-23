# Library to solve the time dependent schrodinger equation in the many-body Fock space.
from pandas import read_excel
import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from qlp.mds.mds_qlpdb import (
    QUBO_to_Ising,
    find_offset,
    AnnealOffset
)

# Get DWave anneal schedule
class s_to_offset():
    def __init__(self):
        params = {"kind":"linear", "fill_value":"extrapolate"} # linear makes for more sensible extrapolation.
        self.anneal_schedule = read_excel(io="./09-1212A-B_DW_2000Q_5_anneal_schedule.xlsx", sheet_name=1)
        self.interpC = interp1d(self.anneal_schedule["s"], self.anneal_schedule["C (normalized)"], **params)
        normA = self.anneal_schedule["A(s) (GHz)"]/self.anneal_schedule["A(s) (GHz)"].max()
        self.interpA = interp1d(self.anneal_schedule["C (normalized)"], normA, **params)
        normB = self.anneal_schedule["B(s) (GHz)"]/self.anneal_schedule["B(s) (GHz)"].max()
        self.interpB = interp1d(self.anneal_schedule["C (normalized)"], normB, **params)
    def sanity_check(self):
        # Sanity check: The data and interpolation should match
        # Interpolate C
        plt.figure()
        ax = plt.axes()
        x = np.linspace(0, 1)
        ax.errorbar(x=self.anneal_schedule["s"], y=self.anneal_schedule["C (normalized)"])
        ax.errorbar(x=x, y=self.interpC(x))
        plt.draw()
        plt.show()
        # Interpolate A and B
        plt.figure()
        ax = plt.axes()
        x = np.linspace(-0.05, 1.05)
        ax.errorbar(x=self.anneal_schedule["s"], y=self.anneal_schedule["A(s) (GHz)"]/self.anneal_schedule["A(s) (GHz)"].max())
        ax.errorbar(x=x, y=self.interpA(self.interpC(x)))
        ax.errorbar(x=self.anneal_schedule["s"], y=self.anneal_schedule["B(s) (GHz)"]/self.anneal_schedule["B(s) (GHz)"].max())
        ax.errorbar(x=x, y=self.interpB(self.interpC(x)))
        plt.draw()
        plt.show()

class AnnealSchedule():
    def __init__(self, offset, h, offset_min, offset_range):
        AO = AnnealOffset(offset)
        self.offset_list, self.offset_tag = AO.fcn(h, offset_min, offset_range)
        self.s2o = s_to_offset()
    def C(self, s):
        C = self.s2o.interpC(s)
        C_offset = C+self.offset_list
        return C_offset
    def A(self, s):
        C = self.C(s)
        return self.s2o.interpA(C)
    def B(self, s):
        C = self.C(s)
        return self.s2o.interpB(C)

def pauli():
    # pauli matrices
    sigx = np.zeros((2, 2))
    sigz = np.zeros((2, 2))
    id2 = np.identity(2)

    sigx[0, 1] = 1.0
    sigx[1, 0] = 1.0
    sigz[0, 0] = 1.0
    sigz[1, 1] = -1.0
    return id2, sigx, sigz

# push local operator to many body Fock space
def pushtoFock(i,local,n):
    fock=np.identity(1)
    for j in range(n):
        if (j==i):
            fock=np.kron(fock,local)
        else:
            fock=np.kron(fock,id2)
    return fock


# hamiltonian (Jij is i>j , i.e., lower diagonal)
def constructIsingH(Jij,hi,n):
    IsingH=np.zeros((2**n,2**n))
    for i in range(n):
        temp=pushtoFock(i,sigz,n)
        IsingH+=hi[i]*temp
        for j in range(i):
            IsingH+=Jij[i,j]*np.dot(temp,pushtoFock(j,sigz,n))
    return IsingH


def constructtransverseH(hxi,n):
    transverseH=np.zeros((2**n,2**n))
    for i in range(n):
        transverseH+=hxi[i]*pushtoFock(i,sigx,n)
    return transverseH

# annealing hamiltonian
def Bij(B):
    return np.asarray([[0.5*(B[i]+B[j]) for i in range(n)] for j in range(n)])

def annealingH(s):
    AxtransverseH=constructtransverseH(AS.A(s)*np.ones((n)),n)
    BxIsingH=constructIsingH(Bij(AS.B(s))*Jij,AS.B(s)*hi,n)
    H=energyscale*(-0.5*AxtransverseH+0.5*BxIsingH)
    return H

# define time-dependent schrodinger equation
def tdse(t,y):
    f=-1j*np.dot(annealingH(t),y)
    return f