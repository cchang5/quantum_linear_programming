"""Models of tdse
"""

# Note: if you want your models to use espressodb features, they must inherit from Base

from django.db import models
from espressodb.base.models import Base
from django.contrib.postgres.fields import JSONField
from django.contrib.postgres.fields import ArrayField


class Tdse(Base):
    tag = models.TextField(null=False, help_text="User-defined tag for easy searches")
    graph = models.ForeignKey(
        "graph.Graph", on_delete=models.CASCADE, help_text=r"Foreign Key to `graph`"
    )

    penalty = models.DecimalField(
        null=False,
        max_digits=6,
        decimal_places=2,
        help_text="Coefficient of penalty term, 0 to 9999.99",
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
        help_text="Wavefunction: type (mixed or pure), temp, initial wavefunction (True or transverse"
    )
    wave_hash = models.TextField(
        null=False, blank=False, help_text="md5 hash for wave parameters"
    )
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
    entropy = ArrayField(
        models.FloatField(null=False),
        null=False,
        help_text="von Neumann entropy vs. time",
    )

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=[
                    "tag",
                    "graph",
                    "penalty",
                    "ising_hash",
                    "offset_hash",
                    "solver_hash",
                    "wave_hash",
                ],
                name="unique_tdse",
            )
        ]
