"""
CLI interface to ml-params-tensorflow. Expected to be bootstrapped by ml-params.
"""

from dataclasses import dataclass
from importlib import import_module
from pkgutil import find_loader

datasets2classes = (
    {}
    if find_loader("ml_prepare") is None
    else getattr(import_module("ml_prepare.datasets"), "datasets2classes")
)


@dataclass
class self(object):
    """
    Simple class to proxy object expected by code generated from `train` function
    :cvar model: The model (probably a `tf.keras.models.Sequential`)
    :cvar data: The data (probably a `tf.data.Dataset`)
    """

    model: object = None
    data: object = None


model = self.model
