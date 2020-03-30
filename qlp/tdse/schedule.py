# pylint: disable=C0103, R0903

"""Functions and classes which define TDSE anneal schedule setups
"""
from numpy import linspace, exp
from scipy.interpolate import interp1d
from pandas import read_excel

from matplotlib.pyplot import subplots, draw, show


class s_to_offset:
    """Class which provides anneal curves for A(s), B(s), C(s) over normalized
    anneal time s in GHz.

    Attributes:
        maxB: float
            Max Dwave magnet field coefficient in GHz.
        anneal_schedule: Dict[str, List[int, int]]
            Boundaries for anneal parameters.
        interpA: np.ndarray
            Interpolated curve for C(s) in GHz.
        interpB: np.ndarray
            Interpolated curve for C(s) in GHz.
        interpC: np.ndarray
            Interpolated curve for C(s) in GHz.
    """

    maxB = 11.8604700

    def __init__(self, fill_value: str, anneal_curve: str):
        """Allocates offset curves A(s), B(s), C(s) over normalized anneal time s in GHz.

        Arguments:
            fill_value: If normalized time (s) is extended beyond (0, 1),
                this option decides whether to extrapolate the anneal schedule,
                or truncate it at the nearest value.
                Options are "extrapolate" or "truncate".
            anneal_curve: Curve normalized time follows.
                Options are "linear", "logistic" or "dwave" (reads in excel file).

        Raises:
            KeyError: If one of the inputs is not reckognized
        """
        if anneal_curve == "linear":

            self.anneal_schedule = {
                "s": [0, 1],
                "C (normalized)": [0, 1],
                "A(s) (GHz)": [self.maxB, 0],
                "B(s) (GHz)": [0, self.maxB],
            }
        elif anneal_curve == "logistic":
            # DWave approx 10 GHz
            s = linspace(0, 1)
            self.anneal_schedule = {
                "s": s,
                "C (normalized)": s,
                "A(s) (GHz)": self.maxB / (1.0 + exp(10.0 * (s - 0.5))),
                "B(s) (GHz)": self.maxB / (1.0 + exp(-10.0 * (s - 0.5))),
            }
        elif anneal_curve == "dwave":
            anneal_schedule = read_excel(
                io="./09-1212A-B_DW_2000Q_5_anneal_schedule.xlsx", sheet_name=1
            )
            self.anneal_schedule = {
                key: anneal_schedule[key].values for key in anneal_schedule.columns
            }
        else:
            raise KeyError(f"Anneal curve {anneal_curve} not reckognized")

        if fill_value == "extrapolate":
            fill_valueA = "extrapolate"
            fill_valueB = "extrapolate"
        elif fill_value == "truncate":
            fill_valueA = (
                self.anneal_schedule["A(s) (GHz)"][0],
                self.anneal_schedule["A(s) (GHz)"][-1],
            )
            fill_valueB = (
                self.anneal_schedule["B(s) (GHz)"][0],
                self.anneal_schedule["B(s) (GHz)"][-1],
            )
        else:
            raise KeyError(f"Fill value {fill_value} not reckognized")

        paramsA = {
            "kind": "linear",
            "bounds_error": False,
            "fill_value": fill_valueA,
        }  # linear makes for more sensible extrapolation.
        paramsB = {
            "kind": "linear",
            "bounds_error": False,
            "fill_value": fill_valueB,
        }  # linear makes for more sensible extrapolation.
        paramsC = {"kind": "linear", "fill_value": "extrapolate"}
        self.interpC = interp1d(
            self.anneal_schedule["s"], self.anneal_schedule["C (normalized)"], **paramsC
        )
        self.interpA = interp1d(
            self.anneal_schedule["C (normalized)"],
            self.anneal_schedule["A(s) (GHz)"],
            **paramsA,
        )
        self.interpB = interp1d(
            self.anneal_schedule["C (normalized)"],
            self.anneal_schedule["B(s) (GHz)"],
            **paramsB,
        )

    def sanity_check(self):
        """Plots anneal schedules for A, B and C

        Sanity check: The data and interpolation should match

        Todo:
            Provide better naming for method.
        """
        # Interpolate C
        _, ax = subplots()
        x = linspace(0, 1)
        ax.errorbar(
            x=self.anneal_schedule["s"], y=self.anneal_schedule["C (normalized)"]
        )
        ax.errorbar(x=x, y=self.interpC(x))
        draw()
        show()
        # Interpolate A and B
        _, ax = subplots()
        x = linspace(-0.05, 1.05)
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
        draw()
        show()
