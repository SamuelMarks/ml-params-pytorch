"""
Type generators. Use these to generate the type annotations throughout this ml-params implementation.

Install doctrans then run, for example:

    python -m doctrans sync_properties \
                       --input-file 'ml_params_pytorch/ml_params/type_generators.py' \
                       --input-eval \
                       --output-file 'ml_params_pytorch/ml_params/trainer.py' \
                       --input-param 'exposed_loss_keys' \
                       --output-param 'TorchTrainer.train.loss' \
                       --input-param 'exposed_activation_keys' \
                       --output-param 'TorchTrainer.train.activation' \
                       --input-param 'exposed_optimizers_keys' \
                       --output-param 'TorchTrainer.train.optimizer' \
                       --input-param 'exposed_optimizer_lr_schedulers' \
                       --output-param 'TorchTrainer.train.lr_scheduler'

"""
from inspect import getmembers
from types import ModuleType
from typing import Any, Dict, Tuple, Union

import torch.nn.modules


def _torch_members(mod: Union[ModuleType, Any], ignore=frozenset()) -> Dict[str, Any]:
    """
    Acquire the members from a torch module (filtered)

    :param mod: The module to run `dir` or `getmembers` against
    :type mod: ```Union[ModuleType, Any]```

    :param ignore: The members to ignore
    :type ignore: ```frozenset```

    :return: members
    :rtype: ```Dict[str, Any]```
    """

    return {
        k: v
        for k, v in getmembers(mod)
        if not k.startswith("_")
        and k not in _global_exclude | ignore
        and v.__class__ is type
    }


_global_exclude: frozenset = frozenset(("Module", "Tensor"))

exposed_activation: Dict[str, Any] = _torch_members(
    torch.nn.modules.activation, frozenset(("Parameter",))
)
exposed_activation_keys: Tuple[str, ...] = tuple(sorted(exposed_activation.keys()))

exposed_losses: Dict[str, Any] = _torch_members(torch.nn.modules.loss)
exposed_loss_keys: Tuple[str, ...] = tuple(sorted(exposed_losses.keys()))

exposed_optimizer_lr_schedulers: Dict[str, Any] = _torch_members(
    torch.optim.lr_scheduler, frozenset(("Optimizer",))
)
exposed_optimizer_lr_schedulers_keys: Tuple[str, ...] = tuple(
    sorted(exposed_optimizer_lr_schedulers.keys())
)

exposed_optimizers: Dict[str, Any] = _torch_members(
    torch.optim, frozenset(("Optimizer",))
)
exposed_optimizers_keys: Tuple[str, ...] = tuple(sorted(exposed_optimizers.keys()))

__all__ = [
    "exposed_activation",
    "exposed_activation_keys",
    "exposed_loss_keys",
    "exposed_losses",
    "exposed_optimizer_lr_schedulers",
    "exposed_optimizer_lr_schedulers_keys",
    "exposed_optimizers",
    "exposed_optimizers_keys",
]