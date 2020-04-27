"""Models of tdse
"""

import numpy as np
from django.db import models
from espressodb.base.models import Base
from django.contrib.postgres.fields import JSONField
from django.contrib.postgres.fields import ArrayField
#from picklefield.fields import PickledObjectField


class Tdse(Base):
    tag = models.TextField(null=False, help_text="User-defined tag for easy searches")
    graph = models.ForeignKey(
        "graph.Graph", on_delete=models.CASCADE, help_text=r"Foreign Key to `graph`"
    )
    ising = JSONField(help_text="Ising parameters: Jij, hi, c, energyscale")
    ising_hash = models.TextField(
        null=False, blank=False, help_text="md5 hash for Ising parameters"
    )
    offset = JSONField(
        help_text="Offset parameters: normalized time, offset, hi_for_offset, offset_min, offset_range, fill_value, anneal_curve"
    )
    offset_hash = models.TextField(
        null=False, blank=False, help_text="md5 hash for offset parameters"
    )
    solver = JSONField(help_text="Solver parameters: method, rtol, atol")
    solver_hash = models.TextField(
        null=False, blank=False, help_text="md5 hash for solver parameters"
    )
    wave = JSONField(
        help_text="Wavefunction: type (mixed or pure), temp, initial wavefunction (True or transverse)"
    )
    wave_hash = models.TextField(
        null=False, blank=False, help_text="md5 hash for wave parameters"
    )
    instance = models.FileField(upload_to="instance/", help_text="save pickled instance of class")
    solution = models.FileField(upload_to="solution/", help_text="save pickled solution")
    time = ArrayField(
        models.FloatField(null=False),
        null=False,
        help_text="Normalized time array for evolution",
    )
    prob = ArrayField(
        models.FloatField(null=False),
        null=False,
        help_text="Ising ground state probability vs. time",
    )
    entropy_params = JSONField(
        help_text="Entropy parameters: nA (number of qubits in partition A), indicesA (einsum index notation), reg"
    )
    entropy_params_hash = models.TextField(
        null=False, blank=False, help_text="md5 hash for entropy parameters"
    )
    entropy = ArrayField(
        models.FloatField(null=False),
        null=False,
        help_text="von Neumann entropy vs. time",
    )

    @property
    def entropy_max(self):
        return np.max(self.entropy)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=[
                    "tag",
                    "graph",
                    "ising_hash",
                    "offset_hash",
                    "solver_hash",
                    "wave_hash",
                    "entropy_params_hash",
                ],
                name="unique_tdse",
            )
        ]
