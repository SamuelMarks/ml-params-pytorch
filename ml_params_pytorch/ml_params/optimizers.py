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
    argument_parser.description = """Implements Averaged Stochastic Gradient Descent.

It has been proposed in `Acceleration of stochastic approximation by
averaging`_.


.. _Acceleration of stochastic approximation by averaging:
    https://dl.acm.org/citation.cfm?id=131098"""
    argument_parser.add_argument(
        "--params",
        help="iterable of parameters to optimize or dicts defining parameter groups",
        required=True,
    )
    argument_parser.add_argument("--lr", type=float, help="learning rate", default=0.01)
    argument_parser.add_argument(
        "--lambd", type=float, help="decay term", default=0.0001
    )
    argument_parser.add_argument(
        "--alpha", type=float, help="power for eta update", default=0.75
    )
    argument_parser.add_argument(
        "--t0", type=float, help="point at which to start averaging", default=1000000.0
    )
    argument_parser.add_argument(
        "--weight_decay", type=int, help="weight decay (L2 penalty)", default=0
    )
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


__ https://arxiv.org/abs/1212.5701"""
    argument_parser.add_argument(
        "--params",
        help="iterable of parameters to optimize or dicts defining parameter groups",
        required=True,
    )
    argument_parser.add_argument(
        "--rho",
        type=float,
        help="coefficient used for computing a running average of squared gradients",
        default=0.9,
    )
    argument_parser.add_argument(
        "--eps",
        type=float,
        help="term added to the denominator to improve numerical stability",
        default=1e-06,
    )
    argument_parser.add_argument(
        "--lr",
        type=float,
        help="coefficient that scale delta before it is applied to the parameters",
        default=1.0,
    )
    argument_parser.add_argument(
        "--weight_decay", type=int, help="weight decay (L2 penalty)", default=0
    )
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


.. _Adaptive Subgradient Methods for Online Learning and Stochastic
    Optimization: http://jmlr.org/papers/v12/duchi11a.html"""
    argument_parser.add_argument(
        "--params",
        help="iterable of parameters to optimize or dicts defining parameter groups",
        required=True,
    )
    argument_parser.add_argument("--lr", type=float, help="learning rate", default=0.01)
    argument_parser.add_argument(
        "--lr_decay", type=int, help="learning rate decay", default=0
    )
    argument_parser.add_argument(
        "--weight_decay", type=int, help="weight decay (L2 penalty)", default=0
    )
    argument_parser.add_argument(
        "--eps",
        type=float,
        help="term added to the denominator to improve numerical stability",
        default=1e-10,
    )
    argument_parser.add_argument("--initial_accumulator_value")
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


.. _Adam\\: A Method for Stochastic Optimization:
    https://arxiv.org/abs/1412.6980
.. _Decoupled Weight Decay Regularization:
    https://arxiv.org/abs/1711.05101
.. _On the Convergence of Adam and Beyond:
    https://openreview.net/forum?id=ryQu7f-RZ"""
    argument_parser.add_argument(
        "--params",
        help="iterable of parameters to optimize or dicts defining parameter groups",
        required=True,
    )
    argument_parser.add_argument(
        "--lr", type=float, help="learning rate", default=0.001
    )
    argument_parser.add_argument(
        "--betas",
        type=loads,
        help="coefficients used for computing running averages of gradient and its square",
        default="[0.9, 0.999]",
    )
    argument_parser.add_argument(
        "--eps",
        type=float,
        help="term added to the denominator to improve numerical stability",
        default=1e-08,
    )
    argument_parser.add_argument(
        "--weight_decay", type=int, help="weight decay (L2 penalty)", default=0
    )
    argument_parser.add_argument(
        "--amsgrad",
        type=bool,
        help="whether to use the AMSGrad variant of this algorithm from the paper `On the Convergence of Adam and Beyond`_",
        default=False,
    )
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


.. _Adam\\: A Method for Stochastic Optimization:
    https://arxiv.org/abs/1412.6980
.. _Decoupled Weight Decay Regularization:
    https://arxiv.org/abs/1711.05101
.. _On the Convergence of Adam and Beyond:
    https://openreview.net/forum?id=ryQu7f-RZ"""
    argument_parser.add_argument(
        "--params",
        help="iterable of parameters to optimize or dicts defining parameter groups",
        required=True,
    )
    argument_parser.add_argument(
        "--lr", type=float, help="learning rate", default=0.001
    )
    argument_parser.add_argument(
        "--betas",
        type=loads,
        help="coefficients used for computing running averages of gradient and its square",
        default="[0.9, 0.999]",
    )
    argument_parser.add_argument(
        "--eps",
        type=float,
        help="term added to the denominator to improve numerical stability",
        default=1e-08,
    )
    argument_parser.add_argument(
        "--weight_decay", type=float, help="weight decay coefficient", default=0.01
    )
    argument_parser.add_argument(
        "--amsgrad",
        type=bool,
        help="whether to use the AMSGrad variant of this algorithm from the paper `On the Convergence of Adam and Beyond`_",
        default=False,
    )
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


__ https://arxiv.org/abs/1412.6980"""
    argument_parser.add_argument(
        "--params",
        help="iterable of parameters to optimize or dicts defining parameter groups",
        required=True,
    )
    argument_parser.add_argument(
        "--lr", type=float, help="learning rate", default=0.002
    )
    argument_parser.add_argument(
        "--betas",
        type=loads,
        help="coefficients used for computing running averages of gradient and its square",
        default="[0.9, 0.999]",
    )
    argument_parser.add_argument(
        "--eps",
        type=float,
        help="term added to the denominator to improve numerical stability",
        default=1e-08,
    )
    argument_parser.add_argument(
        "--weight_decay", type=int, help="weight decay (L2 penalty)", default=0
    )
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
    try reducing the history size, or use a different algorithm."""
    argument_parser.add_argument(
        "--lr", type=int, help="learning rate", required=True, default=1
    )
    argument_parser.add_argument(
        "--max_iter",
        type=int,
        help="maximal number of iterations per optimization step",
        required=True,
        default=20,
    )
    argument_parser.add_argument(
        "--max_eval",
        help="maximal number of function evaluations per optimization step",
        required=True,
        default="max_iter * 1.25).",
    )
    argument_parser.add_argument(
        "--tolerance_grad",
        type=float,
        help="termination tolerance on first order optimality",
        required=True,
        default=1e-07,
    )
    argument_parser.add_argument(
        "--tolerance_change",
        type=float,
        help="termination tolerance on function value/parameter changes",
        required=True,
        default=1e-09,
    )
    argument_parser.add_argument(
        "--history_size",
        type=int,
        help="update history size",
        required=True,
        default=100,
    )
    argument_parser.add_argument(
        "--line_search_fn",
        help="either 'strong_wolfe' or None",
        required=True,
        default="None).",
    )
    argument_parser.add_argument("--params")
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
of the squared gradient."""
    argument_parser.add_argument(
        "--params",
        help="iterable of parameters to optimize or dicts defining parameter groups",
        required=True,
    )
    argument_parser.add_argument("--lr", type=float, help="learning rate", default=0.01)
    argument_parser.add_argument(
        "--momentum", type=int, help="momentum factor", default=0
    )
    argument_parser.add_argument(
        "--alpha", type=float, help="smoothing constant", default=0.99
    )
    argument_parser.add_argument(
        "--eps",
        type=float,
        help="term added to the denominator to improve numerical stability",
        default=1e-08,
    )
    argument_parser.add_argument(
        "--centered",
        type=bool,
        help="if ``True``, compute the centered RMSProp, the gradient is normalized by an estimation of its variance",
        default=False,
    )
    argument_parser.add_argument(
        "--weight_decay", type=int, help="weight decay (L2 penalty)", default=0
    )
    return argument_parser


def RpropConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Implements the resilient backpropagation algorithm."
    argument_parser.add_argument(
        "--params",
        help="iterable of parameters to optimize or dicts defining parameter groups",
        required=True,
    )
    argument_parser.add_argument("--lr", type=float, help="learning rate", default=0.01)
    argument_parser.add_argument(
        "--etas",
        type=loads,
        help="pair of (etaminus, etaplis), that are multiplicative increase and decrease factors",
        default="[0.5, 1.2]",
    )
    argument_parser.add_argument(
        "--step_sizes",
        type=loads,
        help="a pair of minimal and maximal allowed step sizes",
        default="[1e-06, 50]",
    )
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
        help="iterable of parameters to optimize or dicts defining parameter groups",
        required=True,
    )
    argument_parser.add_argument(
        "--lr", type=_RequiredParameter, help="learning rate", required=True
    )
    argument_parser.add_argument(
        "--momentum", type=int, help="momentum factor", default=0
    )
    argument_parser.add_argument(
        "--weight_decay", type=int, help="weight decay (L2 penalty)", default=0
    )
    argument_parser.add_argument(
        "--dampening", type=int, help="dampening for momentum", default=0
    )
    argument_parser.add_argument(
        "--nesterov", type=bool, help="enables Nesterov momentum", default=False
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


.. _Adam\\: A Method for Stochastic Optimization:
    https://arxiv.org/abs/1412.6980"""
    argument_parser.add_argument(
        "--params",
        help="iterable of parameters to optimize or dicts defining parameter groups",
        required=True,
    )
    argument_parser.add_argument(
        "--lr", type=float, help="learning rate", default=0.001
    )
    argument_parser.add_argument(
        "--betas",
        type=loads,
        help="coefficients used for computing running averages of gradient and its square",
        default="[0.9, 0.999]",
    )
    argument_parser.add_argument(
        "--eps",
        type=float,
        help="term added to the denominator to improve numerical stability",
        default=1e-08,
    )
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
