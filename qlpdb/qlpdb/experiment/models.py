"""Models of experiment
"""

# Note: if you want your models to use espressodb features, they must inherit from Base

from django.db import models
from espressodb.base.models import Base
from django.contrib.postgres.fields import JSONField
from django.contrib.postgres.fields import ArrayField


class Experiment(Base):
    graph = models.ForeignKey(
        "graph.Graph", on_delete=models.CASCADE, help_text=r"Foreign Key to `graph`"
    )
    machine = models.TextField(
        null=False, blank=False, help_text="Hardware name (e.g. DW_2000Q_5)"
    )
    settings = JSONField(help_text="Store DWave machine parameters")
    settings_hash = models.TextField(
        null=False, blank=False, help_text="md5 hash of key sorted normalized machine, settings, p, fact dictionary"
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
