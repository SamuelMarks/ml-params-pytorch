"""
CLI interface to ml-params-pytorch. Expected to be bootstrapped by ml-params.
"""

from dataclasses import dataclass


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
