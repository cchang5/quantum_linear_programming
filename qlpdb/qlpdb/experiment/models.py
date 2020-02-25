"""Models of experiment
"""
from typing import List, Dict

from django.db import models
from espressodb.base.models import Base
from django.contrib.postgres.fields import JSONField
from django.contrib.postgres.fields import ArrayField

from django.db.models import Count


class Experiment(Base):
    graph = models.ForeignKey(
        "graph.Graph", on_delete=models.CASCADE, help_text=r"Foreign Key to `graph`"
    )
    machine = models.TextField(
        null=False, blank=False, help_text="Hardware name (e.g. DW_2000Q_5)"
    )
    settings = JSONField(help_text="Store DWave machine parameters")
    settings_hash = models.TextField(
        null=False,
        blank=False,
        help_text="md5 hash of key sorted normalized machine, settings, p, fact dictionary",
    )
    p = models.DecimalField(
        null=False,
        max_digits=6,
        decimal_places=2,
        help_text="Coefficient of penalty term, 0 to 9999.99",
    )
    fact = models.FloatField(null=False, help_text="Manual scaling coefficient")
    qubo = ArrayField(
        ArrayField(models.FloatField(null=False)), help_text="Input QUBO to DWave"
    )

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["graph", "machine", "settings_hash", "p"],
                name="unique_experiment",
            )
        ]

    @property
    def n_data(self) -> int:
        """Returns the number of present data entries
        """
        return self.data_set.count()

    def get_summary(self, n_entries: int = 5) -> List[Dict[str, float]]:
        """Returns a summary of solutions which fulfill the constraints.

        The returned list is sorted according to energy (ascending) and contains the
        energy and number of occurances.

        Arguments:
            n_entries: Number of different energies to be returned.
        """
        exp_data = self.data_set.all()
        satisfied_data = (
            exp_data.filter(constraint_satisfaction=True)
            .values("energy")
            .annotate(occurances=Count("energy"))
        )
        return sorted(list(satisfied_data[:n_entries]), key=lambda el: el["energy"])
