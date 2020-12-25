"""
Config interface to ml-params-tensorflow. Expected to be bootstrapped by ml-params, as well as internally.
"""
from json import loads
from typing import Any


class self(object):
    """
    Simple class to proxy object expected by code generated from `train` function

    :cvar model: The model (probably a `tf.keras.models.Sequential`)
    :cvar data: The data (probably a `tf.data.Dataset`)
    """

    model: Any = None
    data: Any = None


model = self.model


def from_string(cls, s):
    """
    Generate a new object of the class using a loaded s

    :param cls: The class to create
    :type cls: ```Callable[[Any, ...], Any]```

    :param s: The arguments string to be parsed
    :type s: ```str```

    :return: Constructed class
    :rtype: ```Any```
    """
    return cls(**loads(s))
