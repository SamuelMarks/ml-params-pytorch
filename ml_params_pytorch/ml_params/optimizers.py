""" Generated Optimizer CLI parsers """
from yaml import safe_load as loads


def ASGDConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Implements Averaged Stochastic Gradient Descent.\n\nIt has been proposed in `Acceleration of stochastic approximation by\naveraging`_.\n\nArguments:\n    params (iterable): iterable of parameters to optimize or dicts defining\n        parameter groups\n    lr (float, optional): learning rate (default: 1e-2)\n    lambd (float, optional): decay term (default: 1e-4)\n    alpha (float, optional): power for eta update (default: 0.75)\n    t0 (float, optional): point at which to start averaging (default: 1e6)\n    weight_decay (float, optional): weight decay (L2 penalty) (default: 0)\n\n.. _Acceleration of stochastic approximation by averaging:\n    https://dl.acm.org/citation.cfm?id=131098"
    argument_parser.add_argument("--params", help=None, required=True, default=0.01)
    argument_parser.add_argument("--lr", help=None, required=True, default=0.0001)
    argument_parser.add_argument("--lambd", help=None, required=True, default=0.75)
    argument_parser.add_argument("--alpha", help=None, required=True, default=1000000.0)
    argument_parser.add_argument("--t0", help=None, required=True, default=0)
    argument_parser.add_argument("--weight_decay", help=None, required=True)
    return argument_parser


def AdadeltaConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Implements Adadelta algorithm.\n\nIt has been proposed in `ADADELTA: An Adaptive Learning Rate Method`__.\n\nArguments:\n    params (iterable): iterable of parameters to optimize or dicts defining\n        parameter groups\n    rho (float, optional): coefficient used for computing a running average\n        of squared gradients (default: 0.9)\n    eps (float, optional): term added to the denominator to improve\n        numerical stability (default: 1e-6)\n    lr (float, optional): coefficient that scale delta before it is applied\n        to the parameters (default: 1.0)\n    weight_decay (float, optional): weight decay (L2 penalty) (default: 0)\n\n__ https://arxiv.org/abs/1212.5701"
    argument_parser.add_argument("--params", help=None, required=True, default=1.0)
    argument_parser.add_argument("--lr", help=None, required=True, default=0.9)
    argument_parser.add_argument("--rho", help=None, required=True, default=1e-06)
    argument_parser.add_argument("--eps", help=None, required=True, default=0)
    argument_parser.add_argument("--weight_decay", help=None, required=True)
    return argument_parser


def AdagradConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Implements Adagrad algorithm.\n\nIt has been proposed in `Adaptive Subgradient Methods for Online Learning\nand Stochastic Optimization`_.\n\nArguments:\n    params (iterable): iterable of parameters to optimize or dicts defining\n        parameter groups\n    lr (float, optional): learning rate (default: 1e-2)\n    lr_decay (float, optional): learning rate decay (default: 0)\n    weight_decay (float, optional): weight decay (L2 penalty) (default: 0)\n    eps (float, optional): term added to the denominator to improve\n        numerical stability (default: 1e-10)\n\n.. _Adaptive Subgradient Methods for Online Learning and Stochastic\n    Optimization: http://jmlr.org/papers/v12/duchi11a.html"
    argument_parser.add_argument("--params", help=None, required=True, default=0.01)
    argument_parser.add_argument("--lr", help=None, required=True, default=0)
    argument_parser.add_argument("--lr_decay", help=None, required=True, default=0)
    argument_parser.add_argument("--weight_decay", help=None, required=True, default=0)
    argument_parser.add_argument(
        "--initial_accumulator_value", help=None, required=True, default=1e-10
    )
    argument_parser.add_argument("--eps", help=None, required=True)
    return argument_parser


def AdamConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Implements Adam algorithm.\n\nIt has been proposed in `Adam: A Method for Stochastic Optimization`_.\nThe implementation of the L2 penalty follows changes proposed in\n`Decoupled Weight Decay Regularization`_.\n\nArguments:\n    params (iterable): iterable of parameters to optimize or dicts defining\n        parameter groups\n    lr (float, optional): learning rate (default: 1e-3)\n    betas (Tuple[float, float], optional): coefficients used for computing\n        running averages of gradient and its square (default: (0.9, 0.999))\n    eps (float, optional): term added to the denominator to improve\n        numerical stability (default: 1e-8)\n    weight_decay (float, optional): weight decay (L2 penalty) (default: 0)\n    amsgrad (boolean, optional): whether to use the AMSGrad variant of this\n        algorithm from the paper `On the Convergence of Adam and Beyond`_\n        (default: False)\n\n.. _Adam\\: A Method for Stochastic Optimization:\n    https://arxiv.org/abs/1412.6980\n.. _Decoupled Weight Decay Regularization:\n    https://arxiv.org/abs/1711.05101\n.. _On the Convergence of Adam and Beyond:\n    https://openreview.net/forum?id=ryQu7f-RZ"
    argument_parser.add_argument("--params", help=None, required=True, default=0.001)
    argument_parser.add_argument(
        "--lr", type=loads, help=None, required=True, default="(0.9, 0.999)"
    )
    argument_parser.add_argument("--betas", help=None, required=True, default=1e-08)
    argument_parser.add_argument("--eps", help=None, required=True, default=0)
    argument_parser.add_argument(
        "--weight_decay", help=None, required=True, default=False
    )
    argument_parser.add_argument("--amsgrad", help=None, required=True)
    return argument_parser


def AdamWConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Implements AdamW algorithm.\n\nThe original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.\nThe AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.\n\nArguments:\n    params (iterable): iterable of parameters to optimize or dicts defining\n        parameter groups\n    lr (float, optional): learning rate (default: 1e-3)\n    betas (Tuple[float, float], optional): coefficients used for computing\n        running averages of gradient and its square (default: (0.9, 0.999))\n    eps (float, optional): term added to the denominator to improve\n        numerical stability (default: 1e-8)\n    weight_decay (float, optional): weight decay coefficient (default: 1e-2)\n    amsgrad (boolean, optional): whether to use the AMSGrad variant of this\n        algorithm from the paper `On the Convergence of Adam and Beyond`_\n        (default: False)\n\n.. _Adam\\: A Method for Stochastic Optimization:\n    https://arxiv.org/abs/1412.6980\n.. _Decoupled Weight Decay Regularization:\n    https://arxiv.org/abs/1711.05101\n.. _On the Convergence of Adam and Beyond:\n    https://openreview.net/forum?id=ryQu7f-RZ"
    argument_parser.add_argument("--params", help=None, required=True, default=0.001)
    argument_parser.add_argument(
        "--lr", type=loads, help=None, required=True, default="(0.9, 0.999)"
    )
    argument_parser.add_argument("--betas", help=None, required=True, default=1e-08)
    argument_parser.add_argument("--eps", help=None, required=True, default=0.01)
    argument_parser.add_argument(
        "--weight_decay", help=None, required=True, default=False
    )
    argument_parser.add_argument("--amsgrad", help=None, required=True)
    return argument_parser


def AdamaxConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Implements Adamax algorithm (a variant of Adam based on infinity norm).\n\nIt has been proposed in `Adam: A Method for Stochastic Optimization`__.\n\nArguments:\n    params (iterable): iterable of parameters to optimize or dicts defining\n        parameter groups\n    lr (float, optional): learning rate (default: 2e-3)\n    betas (Tuple[float, float], optional): coefficients used for computing\n        running averages of gradient and its square\n    eps (float, optional): term added to the denominator to improve\n        numerical stability (default: 1e-8)\n    weight_decay (float, optional): weight decay (L2 penalty) (default: 0)\n\n__ https://arxiv.org/abs/1412.6980"
    argument_parser.add_argument("--params", help=None, required=True, default=0.002)
    argument_parser.add_argument(
        "--lr", type=loads, help=None, required=True, default="(0.9, 0.999)"
    )
    argument_parser.add_argument("--betas", help=None, required=True, default=1e-08)
    argument_parser.add_argument("--eps", help=None, required=True, default=0)
    argument_parser.add_argument("--weight_decay", help=None, required=True)
    return argument_parser


def LBFGSConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Implements L-BFGS algorithm, heavily inspired by `minFunc\n<https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html>`.\n\n.. warning::\n    This optimizer doesn't support per-parameter options and parameter\n    groups (there can be only one).\n\n.. warning::\n    Right now all parameters have to be on a single device. This will be\n    improved in the future.\n\n.. note::\n    This is a very memory intensive optimizer (it requires additional\n    ``param_bytes * (history_size + 1)`` bytes). If it doesn't fit in memory\n    try reducing the history size, or use a different algorithm.\n\nArguments:\n    lr (float): learning rate (default: 1)\n    max_iter (int): maximal number of iterations per optimization step\n        (default: 20)\n    max_eval (int): maximal number of function evaluations per optimization\n        step (default: max_iter * 1.25).\n    tolerance_grad (float): termination tolerance on first order optimality\n        (default: 1e-5).\n    tolerance_change (float): termination tolerance on function\n        value/parameter changes (default: 1e-9).\n    history_size (int): update history size (default: 100).\n    line_search_fn (str): either 'strong_wolfe' or None (default: None)."
    argument_parser.add_argument("--params", help=None, required=True, default=1)
    argument_parser.add_argument("--lr", help=None, required=True, default=20)
    argument_parser.add_argument("--max_iter", help=None)
    argument_parser.add_argument("--max_eval", help=None, required=True, default=1e-07)
    argument_parser.add_argument(
        "--tolerance_grad", help=None, required=True, default=1e-09
    )
    argument_parser.add_argument(
        "--tolerance_change", help=None, required=True, default=100
    )
    argument_parser.add_argument("--history_size", help=None)
    argument_parser.add_argument("--line_search_fn", help=None, required=True)
    return argument_parser


def RMSpropConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Implements RMSprop algorithm.\n\nProposed by G. Hinton in his\n`course <https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_.\n\nThe centered version first appears in `Generating Sequences\nWith Recurrent Neural Networks <https://arxiv.org/pdf/1308.0850v5.pdf>`_.\n\nThe implementation here takes the square root of the gradient average before\nadding epsilon (note that TensorFlow interchanges these two operations). The effective\nlearning rate is thus :math:`\\alpha/(\\sqrt{v} + \\epsilon)` where :math:`\\alpha`\nis the scheduled learning rate and :math:`v` is the weighted moving average\nof the squared gradient.\n\nArguments:\n    params (iterable): iterable of parameters to optimize or dicts defining\n        parameter groups\n    lr (float, optional): learning rate (default: 1e-2)\n    momentum (float, optional): momentum factor (default: 0)\n    alpha (float, optional): smoothing constant (default: 0.99)\n    eps (float, optional): term added to the denominator to improve\n        numerical stability (default: 1e-8)\n    centered (bool, optional) : if ``True``, compute the centered RMSProp,\n        the gradient is normalized by an estimation of its variance\n    weight_decay (float, optional): weight decay (L2 penalty) (default: 0)"
    argument_parser.add_argument("--params", help=None, required=True, default=0.01)
    argument_parser.add_argument("--lr", help=None, required=True, default=0.99)
    argument_parser.add_argument("--alpha", help=None, required=True, default=1e-08)
    argument_parser.add_argument("--eps", help=None, required=True, default=0)
    argument_parser.add_argument("--weight_decay", help=None, required=True, default=0)
    argument_parser.add_argument("--momentum", help=None, required=True, default=False)
    argument_parser.add_argument("--centered", help=None, required=True)
    return argument_parser


def RpropConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Implements the resilient backpropagation algorithm.\n\nArguments:\n    params (iterable): iterable of parameters to optimize or dicts defining\n        parameter groups\n    lr (float, optional): learning rate (default: 1e-2)\n    etas (Tuple[float, float], optional): pair of (etaminus, etaplis), that\n        are multiplicative increase and decrease factors\n        (default: (0.5, 1.2))\n    step_sizes (Tuple[float, float], optional): a pair of minimal and\n        maximal allowed step sizes (default: (1e-6, 50))"
    argument_parser.add_argument("--params", help=None, required=True, default=0.01)
    argument_parser.add_argument(
        "--lr", type=loads, help=None, required=True, default="(0.5, 1.2)"
    )
    argument_parser.add_argument(
        "--etas", type=loads, help=None, required=True, default="(1e-06, 50)"
    )
    argument_parser.add_argument("--step_sizes", help=None, required=True)
    return argument_parser


def SGDConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Implements stochastic gradient descent (optionally with momentum).\n\nNesterov momentum is based on the formula from\n`On the importance of initialization and momentum in deep learning`__.\n\n\nExample:\n    >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)\n    >>> optimizer.zero_grad()\n    >>> loss_fn(model(input), target).backward()\n    >>> optimizer.step()\n\n__ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf\n\n.. note::\n    The implementation of SGD with Momentum/Nesterov subtly differs from\n    Sutskever et. al. and implementations in some other frameworks.\n\n    Considering the specific case of Momentum, the update can be written as\n\n    .. math::\n        \\begin{aligned}\n            v_{t+1} & = \\mu * v_{t} + g_{t+1}, \\\\\n            p_{t+1} & = p_{t} - \\text{lr} * v_{t+1},\n        \\end{aligned}\n\n    where :math:`p`, :math:`g`, :math:`v` and :math:`\\mu` denote the \n    parameters, gradient, velocity, and momentum respectively.\n\n    This is in contrast to Sutskever et. al. and\n    other frameworks which employ an update of the form\n\n    .. math::\n        \\begin{aligned}\n            v_{t+1} & = \\mu * v_{t} + \\text{lr} * g_{t+1}, \\\\\n            p_{t+1} & = p_{t} - v_{t+1}.\n        \\end{aligned}\n\n    The Nesterov version is analogously modified."
    argument_parser.add_argument(
        "--params",
        type=str,
        help="iterable of parameters to optimize or dicts defining\n        parameter groups",
        required=True,
        default="required",
    )
    argument_parser.add_argument(
        "--lr", type=float, help="learning rate", required=True
    )
    argument_parser.add_argument(
        "--momentum", type=float, help="momentum factor ", default=0
    )
    argument_parser.add_argument(
        "--weight_decay", type=float, help="weight decay (L2 penalty) ", default=0
    )
    argument_parser.add_argument(
        "--dampening", type=float, help="dampening for momentum ", default=0
    )
    argument_parser.add_argument(
        "--nesterov", type=bool, help="enables Nesterov momentum ", default=False
    )
    return argument_parser


def SparseAdamConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Implements lazy version of Adam algorithm suitable for sparse tensors.\n\nIn this variant, only moments that show up in the gradient get updated, and\nonly those portions of the gradient get applied to the parameters.\n\nArguments:\n    params (iterable): iterable of parameters to optimize or dicts defining\n        parameter groups\n    lr (float, optional): learning rate (default: 1e-3)\n    betas (Tuple[float, float], optional): coefficients used for computing\n        running averages of gradient and its square (default: (0.9, 0.999))\n    eps (float, optional): term added to the denominator to improve\n        numerical stability (default: 1e-8)\n\n.. _Adam\\: A Method for Stochastic Optimization:\n    https://arxiv.org/abs/1412.6980"
    argument_parser.add_argument("--params", help=None, required=True, default=0.001)
    argument_parser.add_argument(
        "--lr", type=loads, help=None, required=True, default="(0.9, 0.999)"
    )
    argument_parser.add_argument("--betas", help=None, required=True, default=1e-08)
    argument_parser.add_argument("--eps", help=None, required=True)
    return argument_parser


__all__ = [
    "ASGDConfig",
    "AdadeltaConfig",
    "AdagradConfig",
    "AdamConfig",
    "AdamWConfig",
    "AdamaxConfig",
    "LBFGSConfig",
    "RMSpropConfig",
    "RpropConfig",
    "SGDConfig",
    "SparseAdamConfig",
]
