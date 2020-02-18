"""Prepares the usage of the qlpdb module
"""
import os
from django import setup as _setup


def _init():
    """Initializes the django environment for qlpdb
    """
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "qlpdb.config.settings")
    _setup()


_init()
