""" Generated Optimizer CLI parsers """
from yaml import safe_load as loads

NoneType = type(None)


def ASGDConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Implements Averaged Stochastic Gradient Descent.

It has been proposed in `Acceleration of stochastic approximation by
averaging`_.

Arguments:
    params (iterable): iterable of parameters to optimize or dicts defining
        parameter groups
    lr (float, optional): learning rate (default: 1e-2)
    lambd (float, optional): decay term (default: 1e-4)
    alpha (float, optional): power for eta update (default: 0.75)
    t0 (float, optional): point at which to start averaging (default: 1e6)
    weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

.. _Acceleration of stochastic approximation by averaging:
    https://dl.acm.org/citation.cfm?id=131098"""
    argument_parser.add_argument("--params", type=float, required=True, default=0.01)
    argument_parser.add_argument("--lr", type=float, required=True, default=0.0001)
    argument_parser.add_argument("--lambd", type=float, required=True, default=0.75)
    argument_parser.add_argument(
        "--alpha", type=float, required=True, default=1000000.0
    )
    argument_parser.add_argument("--t0", required=True, default=0)
    argument_parser.add_argument("--weight_decay", required=True)
    return argument_parser


def AdadeltaConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Implements Adadelta algorithm.

It has been proposed in `ADADELTA: An Adaptive Learning Rate Method`__.

Arguments:
    params (iterable): iterable of parameters to optimize or dicts defining
        parameter groups
    rho (float, optional): coefficient used for computing a running average
        of squared gradients (default: 0.9)
    eps (float, optional): term added to the denominator to improve
        numerical stability (default: 1e-6)
    lr (float, optional): coefficient that scale delta before it is applied
        to the parameters (default: 1.0)
    weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

__ https://arxiv.org/abs/1212.5701"""
    argument_parser.add_argument("--params", type=float, required=True, default=1.0)
    argument_parser.add_argument("--lr", type=float, required=True, default=0.9)
    argument_parser.add_argument("--rho", type=float, required=True, default=1e-06)
    argument_parser.add_argument("--eps", required=True, default=0)
    argument_parser.add_argument("--weight_decay", required=True)
    return argument_parser


def AdagradConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Implements Adagrad algorithm.

It has been proposed in `Adaptive Subgradient Methods for Online Learning
and Stochastic Optimization`_.

Arguments:
    params (iterable): iterable of parameters to optimize or dicts defining
        parameter groups
    lr (float, optional): learning rate (default: 1e-2)
    lr_decay (float, optional): learning rate decay (default: 0)
    weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    eps (float, optional): term added to the denominator to improve
        numerical stability (default: 1e-10)

.. _Adaptive Subgradient Methods for Online Learning and Stochastic
    Optimization: http://jmlr.org/papers/v12/duchi11a.html"""
    argument_parser.add_argument("--params", type=float, required=True, default=0.01)
    argument_parser.add_argument("--lr", required=True, default=0)
    argument_parser.add_argument("--lr_decay", required=True, default=0)
    argument_parser.add_argument("--weight_decay", required=True, default=0)
    argument_parser.add_argument(
        "--initial_accumulator_value", type=float, required=True, default=1e-10
    )
    argument_parser.add_argument("--eps", required=True)
    return argument_parser


def AdamConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Implements Adam algorithm.

It has been proposed in `Adam: A Method for Stochastic Optimization`_.
The implementation of the L2 penalty follows changes proposed in
`Decoupled Weight Decay Regularization`_.

Arguments:
    params (iterable): iterable of parameters to optimize or dicts defining
        parameter groups
    lr (float, optional): learning rate (default: 1e-3)
    betas (Tuple[float, float], optional): coefficients used for computing
        running averages of gradient and its square (default: (0.9, 0.999))
    eps (float, optional): term added to the denominator to improve
        numerical stability (default: 1e-8)
    weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    amsgrad (boolean, optional): whether to use the AMSGrad variant of this
        algorithm from the paper `On the Convergence of Adam and Beyond`_
        (default: False)

.. _Adam\\: A Method for Stochastic Optimization:
    https://arxiv.org/abs/1412.6980
.. _Decoupled Weight Decay Regularization:
    https://arxiv.org/abs/1711.05101
.. _On the Convergence of Adam and Beyond:
    https://openreview.net/forum?id=ryQu7f-RZ"""
    argument_parser.add_argument("--params", type=float, required=True, default=0.001)
    argument_parser.add_argument(
        "--lr", type=loads, required=True, default="(0.9, 0.999)"
    )
    argument_parser.add_argument("--betas", type=float, required=True, default=1e-08)
    argument_parser.add_argument("--eps", required=True, default=0)
    argument_parser.add_argument("--weight_decay", required=True, default=False)
    argument_parser.add_argument("--amsgrad", required=True)
    return argument_parser


def AdamWConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Implements AdamW algorithm.

The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.

Arguments:
    params (iterable): iterable of parameters to optimize or dicts defining
        parameter groups
    lr (float, optional): learning rate (default: 1e-3)
    betas (Tuple[float, float], optional): coefficients used for computing
        running averages of gradient and its square (default: (0.9, 0.999))
    eps (float, optional): term added to the denominator to improve
        numerical stability (default: 1e-8)
    weight_decay (float, optional): weight decay coefficient (default: 1e-2)
    amsgrad (boolean, optional): whether to use the AMSGrad variant of this
        algorithm from the paper `On the Convergence of Adam and Beyond`_
        (default: False)

.. _Adam\\: A Method for Stochastic Optimization:
    https://arxiv.org/abs/1412.6980
.. _Decoupled Weight Decay Regularization:
    https://arxiv.org/abs/1711.05101
.. _On the Convergence of Adam and Beyond:
    https://openreview.net/forum?id=ryQu7f-RZ"""
    argument_parser.add_argument("--params", type=float, required=True, default=0.001)
    argument_parser.add_argument(
        "--lr", type=loads, required=True, default="(0.9, 0.999)"
    )
    argument_parser.add_argument("--betas", type=float, required=True, default=1e-08)
    argument_parser.add_argument("--eps", type=float, required=True, default=0.01)
    argument_parser.add_argument("--weight_decay", required=True, default=False)
    argument_parser.add_argument("--amsgrad", required=True)
    return argument_parser


def AdamaxConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Implements Adamax algorithm (a variant of Adam based on infinity norm).

It has been proposed in `Adam: A Method for Stochastic Optimization`__.

Arguments:
    params (iterable): iterable of parameters to optimize or dicts defining
        parameter groups
    lr (float, optional): learning rate (default: 2e-3)
    betas (Tuple[float, float], optional): coefficients used for computing
        running averages of gradient and its square
    eps (float, optional): term added to the denominator to improve
        numerical stability (default: 1e-8)
    weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

__ https://arxiv.org/abs/1412.6980"""
    argument_parser.add_argument("--params", type=float, required=True, default=0.002)
    argument_parser.add_argument(
        "--lr", type=loads, required=True, default="(0.9, 0.999)"
    )
    argument_parser.add_argument("--betas", type=float, required=True, default=1e-08)
    argument_parser.add_argument("--eps", required=True, default=0)
    argument_parser.add_argument("--weight_decay", required=True)
    return argument_parser


def LBFGSConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Implements L-BFGS algorithm, heavily inspired by `minFunc
<https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html>`.

.. warning::
    This optimizer doesn't support per-parameter options and parameter
    groups (there can be only one).

.. warning::
    Right now all parameters have to be on a single device. This will be
    improved in the future.

.. note::
    This is a very memory intensive optimizer (it requires additional
    ``param_bytes * (history_size + 1)`` bytes). If it doesn't fit in memory
    try reducing the history size, or use a different algorithm.

Arguments:
    lr (float): learning rate (default: 1)
    max_iter (int): maximal number of iterations per optimization step
        (default: 20)
    max_eval (int): maximal number of function evaluations per optimization
        step (default: max_iter * 1.25).
    tolerance_grad (float): termination tolerance on first order optimality
        (default: 1e-5).
    tolerance_change (float): termination tolerance on function
        value/parameter changes (default: 1e-9).
    history_size (int): update history size (default: 100).
    line_search_fn (str): either 'strong_wolfe' or None (default: None)."""
    argument_parser.add_argument("--params", type=int, required=True, default=1)
    argument_parser.add_argument("--lr", type=int, required=True, default=20)
    argument_parser.add_argument("--max_iter", required=True)
    argument_parser.add_argument("--max_eval", type=float, required=True, default=1e-07)
    argument_parser.add_argument(
        "--tolerance_grad", type=float, required=True, default=1e-09
    )
    argument_parser.add_argument(
        "--tolerance_change", type=int, required=True, default=100
    )
    argument_parser.add_argument("--history_size", required=True)
    argument_parser.add_argument("--line_search_fn", required=True)
    return argument_parser


def RMSpropConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Implements RMSprop algorithm.

Proposed by G. Hinton in his
`course <https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_.

The centered version first appears in `Generating Sequences
With Recurrent Neural Networks <https://arxiv.org/pdf/1308.0850v5.pdf>`_.

The implementation here takes the square root of the gradient average before
adding epsilon (note that TensorFlow interchanges these two operations). The effective
learning rate is thus :math:`\\alpha/(\\sqrt{v} + \\epsilon)` where :math:`\\alpha`
is the scheduled learning rate and :math:`v` is the weighted moving average
of the squared gradient.

Arguments:
    params (iterable): iterable of parameters to optimize or dicts defining
        parameter groups
    lr (float, optional): learning rate (default: 1e-2)
    momentum (float, optional): momentum factor (default: 0)
    alpha (float, optional): smoothing constant (default: 0.99)
    eps (float, optional): term added to the denominator to improve
        numerical stability (default: 1e-8)
    centered (bool, optional) : if ``True``, compute the centered RMSProp,
        the gradient is normalized by an estimation of its variance
    weight_decay (float, optional): weight decay (L2 penalty) (default: 0)"""
    argument_parser.add_argument("--params", type=float, required=True, default=0.01)
    argument_parser.add_argument("--lr", type=float, required=True, default=0.99)
    argument_parser.add_argument("--alpha", type=float, required=True, default=1e-08)
    argument_parser.add_argument("--eps", required=True, default=0)
    argument_parser.add_argument("--weight_decay", required=True, default=0)
    argument_parser.add_argument("--momentum", required=True, default=False)
    argument_parser.add_argument("--centered", required=True)
    return argument_parser


def RpropConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Implements the resilient backpropagation algorithm.

Arguments:
    params (iterable): iterable of parameters to optimize or dicts defining
        parameter groups
    lr (float, optional): learning rate (default: 1e-2)
    etas (Tuple[float, float], optional): pair of (etaminus, etaplis), that
        are multiplicative increase and decrease factors
        (default: (0.5, 1.2))
    step_sizes (Tuple[float, float], optional): a pair of minimal and
        maximal allowed step sizes (default: (1e-6, 50))"""
    argument_parser.add_argument("--params", type=float, required=True, default=0.01)
    argument_parser.add_argument(
        "--lr", type=loads, required=True, default="(0.5, 1.2)"
    )
    argument_parser.add_argument(
        "--etas", type=loads, required=True, default="(1e-06, 50)"
    )
    argument_parser.add_argument("--step_sizes", required=True)
    return argument_parser


def SGDConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Implements stochastic gradient descent (optionally with momentum).

Nesterov momentum is based on the formula from
`On the importance of initialization and momentum in deep learning`__.


Example:
    >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    >>> optimizer.zero_grad()
    >>> loss_fn(model(input), target).backward()
    >>> optimizer.step()

__ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

.. note::
    The implementation of SGD with Momentum/Nesterov subtly differs from
    Sutskever et. al. and implementations in some other frameworks.

    Considering the specific case of Momentum, the update can be written as

    .. math::
        \\begin{aligned}
            v_{t+1} & = \\mu * v_{t} + g_{t+1}, \\\\
            p_{t+1} & = p_{t} - \\text{lr} * v_{t+1},
        \\end{aligned}

    where :math:`p`, :math:`g`, :math:`v` and :math:`\\mu` denote the 
    parameters, gradient, velocity, and momentum respectively.

    This is in contrast to Sutskever et. al. and
    other frameworks which employ an update of the form

    .. math::
        \\begin{aligned}
            v_{t+1} & = \\mu * v_{t} + \\text{lr} * g_{t+1}, \\\\
            p_{t+1} & = p_{t} - v_{t+1}.
        \\end{aligned}

    The Nesterov version is analogously modified."""
    argument_parser.add_argument(
        "--params",
        type=str,
        help="""iterable of parameters to optimize or dicts defining
        parameter groups""",
        required=True,
        default="required",
    )
    argument_parser.add_argument(
        "--lr", type=float, help="learning rate", required=True
    )
    argument_parser.add_argument(
        "--momentum", type=int, help="momentum factor ", default=0
    )
    argument_parser.add_argument(
        "--weight_decay", type=int, help="weight decay (L2 penalty) ", default=0
    )
    argument_parser.add_argument(
        "--dampening", type=int, help="dampening for momentum ", default=0
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
    argument_parser.description = """Implements lazy version of Adam algorithm suitable for sparse tensors.

In this variant, only moments that show up in the gradient get updated, and
only those portions of the gradient get applied to the parameters.

Arguments:
    params (iterable): iterable of parameters to optimize or dicts defining
        parameter groups
    lr (float, optional): learning rate (default: 1e-3)
    betas (Tuple[float, float], optional): coefficients used for computing
        running averages of gradient and its square (default: (0.9, 0.999))
    eps (float, optional): term added to the denominator to improve
        numerical stability (default: 1e-8)

.. _Adam\\: A Method for Stochastic Optimization:
    https://arxiv.org/abs/1412.6980"""
    argument_parser.add_argument("--params", type=float, required=True, default=0.001)
    argument_parser.add_argument(
        "--lr", type=loads, required=True, default="(0.9, 0.999)"
    )
    argument_parser.add_argument("--betas", type=float, required=True, default=1e-08)
    argument_parser.add_argument("--eps", required=True)
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
