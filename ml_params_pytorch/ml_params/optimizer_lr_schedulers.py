""" Generated Optimizer_lr_scheduler CLI parsers """


def CosineAnnealingLRConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :returns: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Set the learning rate of each parameter group using a cosine annealing
schedule, where :math:`\\eta_{max}` is set to the initial lr and
:math:`T_{cur}` is the number of epochs since the last restart in SGDR:

.. math::
    \\begin{aligned}
        \\eta_t & = \\eta_{min} + \\frac{1}{2}(\\eta_{max} - \\eta_{min})\\left(1
        + \\cos\\left(\\frac{T_{cur}}{T_{max}}\\pi\\right)\\right),
        & T_{cur} \\neq (2k+1)T_{max}; \\\\
        \\eta_{t+1} & = \\eta_{t} + \\frac{1}{2}(\\eta_{max} - \\eta_{min})
        \\left(1 - \\cos\\left(\\frac{1}{T_{max}}\\pi\\right)\\right),
        & T_{cur} = (2k+1)T_{max}.
    \\end{aligned}

When last_epoch=-1, sets initial lr as lr. Notice that because the schedule
is defined recursively, the learning rate can be simultaneously modified
outside this scheduler by other operators. If the learning rate is set
solely by this scheduler, the learning rate at each step becomes:

.. math::
    \\eta_t = \\eta_{min} + \\frac{1}{2}(\\eta_{max} - \\eta_{min})\\left(1 +
    \\cos\\left(\\frac{T_{cur}}{T_{max}}\\pi\\right)\\right)

It has been proposed in
`SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
implements the cosine annealing part of SGDR, and not the restarts.


.. _SGDR\\: Stochastic Gradient Descent with Warm Restarts:
    https://arxiv.org/abs/1608.03983"""
    argument_parser.add_argument(
        "--optimizer", help="Wrapped optimizer.", required=True
    )
    argument_parser.add_argument(
        "--T_max", type=int, help="Maximum number of iterations.", required=True
    )
    argument_parser.add_argument(
        "--eta_min", type=int, help="Minimum learning rate.", required=True, default=0
    )
    argument_parser.add_argument(
        "--last_epoch",
        type=int,
        help="The index of last epoch.",
        required=True,
        default=-1,
    )
    argument_parser.add_argument(
        "--verbose",
        type=bool,
        help="If ``True``, prints a message to stdout for each update.",
        required=True,
        default=False,
    )
    return argument_parser


def CosineAnnealingWarmRestartsConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :returns: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Set the learning rate of each parameter group using a cosine annealing
schedule, where :math:`\\eta_{max}` is set to the initial lr, :math:`T_{cur}`
is the number of epochs since the last restart and :math:`T_{i}` is the number
of epochs between two warm restarts in SGDR:

.. math::
    \\eta_t = \\eta_{min} + \\frac{1}{2}(\\eta_{max} - \\eta_{min})\\left(1 +
    \\cos\\left(\\frac{T_{cur}}{T_{i}}\\pi\\right)\\right)

When :math:`T_{cur}=T_{i}`, set :math:`\\eta_t = \\eta_{min}`.
When :math:`T_{cur}=0` after restart, set :math:`\\eta_t=\\eta_{max}`.

It has been proposed in
`SGDR: Stochastic Gradient Descent with Warm Restarts`_.


.. _SGDR\\: Stochastic Gradient Descent with Warm Restarts:
    https://arxiv.org/abs/1608.03983"""
    argument_parser.add_argument(
        "--optimizer", help="Wrapped optimizer.", required=True
    )
    argument_parser.add_argument(
        "--T_0",
        type=int,
        help="Number of iterations for the first restart.",
        required=True,
    )
    argument_parser.add_argument(
        "--T_mult",
        type=int,
        help="A factor increases :math:`T_{i}` after a restart.",
        default=1,
    )
    argument_parser.add_argument(
        "--eta_min", type=int, help="Minimum learning rate.", default=0
    )
    argument_parser.add_argument(
        "--last_epoch", type=int, help="The index of last epoch.", default=-1
    )
    argument_parser.add_argument(
        "--verbose",
        type=bool,
        help="If ``True``, prints a message to stdout for each update.",
        required=True,
        default=False,
    )
    return argument_parser


def CounterConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :returns: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Dict subclass for counting hashable items.  Sometimes called a bag
or multiset.  Elements are stored as dictionary keys and their counts
are stored as dictionary values.

>>> c = Counter('abcdeabcdabcaba')  # count elements from a string

>>> c.most_common(3)                # three most common elements
[('a', 5), ('b', 4), ('c', 3)]
>>> sorted(c)                       # list all unique elements
['a', 'b', 'c', 'd', 'e']
>>> ''.join(sorted(c.elements()))   # list elements with repetitions
'aaaaabbbbcccdde'
>>> sum(c.values())                 # total of all counts
15

>>> c['a']                          # count of letter 'a'
5
>>> for elem in 'shazam':           # update counts from an iterable
...     c[elem] += 1                # by adding 1 to each element's count
>>> c['a']                          # now there are seven 'a'
7
>>> del c['b']                      # remove all 'b'
>>> c['b']                          # now there are zero 'b'
0

>>> d = Counter('simsalabim')       # make another counter
>>> c.update(d)                     # add in the second counter
>>> c['a']                          # now there are nine 'a'
9

>>> c.clear()                       # empty the counter
>>> c
Counter()

Note:  If a count is set to zero or reduced to zero, it will remain
in the counter until the entry is deleted or the counter is cleared:

>>> c = Counter('aaabbc')
>>> c['b'] -= 2                     # reduce the count of 'b' by two
>>> c.most_common()                 # 'b' is still in, but its count is zero
[('a', 3), ('c', 1), ('b', 0)]"""
    return argument_parser


def CyclicLRConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :returns: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Sets the learning rate of each parameter group according to
cyclical learning rate policy (CLR). The policy cycles the learning
rate between two boundaries with a constant frequency, as detailed in
the paper `Cyclical Learning Rates for Training Neural Networks`_.
The distance between the two boundaries can be scaled on a per-iteration
or per-cycle basis.

Cyclical learning rate policy changes the learning rate after every batch.
`step` should be called after a batch has been used for training.

This class has three built-in policies, as put forth in the paper:

* "triangular": A basic triangular cycle without amplitude scaling.
* "triangular2": A basic triangular cycle that scales initial amplitude by half each cycle.
* "exp_range": A cycle that scales initial amplitude by :math:`\\text{gamma}^{\\text{cycle iterations}}`
  at each cycle iteration.

This implementation was adapted from the github repo: `bckenstler/CLR`_


    gamma (float): Constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
            Default: 1.0
        scale_fn (function): Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            If specified, then 'mode' is ignored.
            Default: None
        scale_mode (str): {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle).
            Default: 'cycle'
        cycle_momentum (bool): If ``True``, momentum is cycled inversely
            to learning rate between 'base_momentum' and 'max_momentum'.
            Default: True
        base_momentum (float or list): Lower momentum boundaries in the cycle
            for each parameter group. Note that momentum is cycled inversely
            to learning rate; at the peak of a cycle, momentum is
            'base_momentum' and learning rate is 'max_lr'.
            Default: 0.8
        max_momentum (float or list): Upper momentum boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_momentum - base_momentum).
            The momentum at any cycle is the difference of max_momentum
            and some scaling of the amplitude; therefore
            base_momentum may not actually be reached depending on
            scaling function. Note that momentum is cycled inversely
            to learning rate; at the start of a cycle, momentum is 'max_momentum'
            and learning rate is 'base_lr'
            Default: 0.9
        last_epoch (int): The index of the last batch. This parameter is used when
            resuming a training job. Since `step()` should be invoked after each
            batch instead of after each epoch, this number represents the total
            number of *batches* computed, not the total number of epochs computed.
            When last_epoch=-1, the schedule is started from the beginning.
            Default: -1
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.


Example:
    >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    >>> scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)
    >>> data_loader = torch.utils.data.DataLoader(...)
    >>> for epoch in range(10):
    >>>     for batch in data_loader:
    >>>         train_batch(...)
    >>>         scheduler.step()


.. _Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
.. _bckenstler/CLR: https://github.com/bckenstler/CLR"""
    argument_parser.add_argument(
        "--optimizer", help="Wrapped optimizer.", required=True
    )
    argument_parser.add_argument(
        "--base_lr",
        help="Initial learning rate which is the lower boundary in the cycle for each parameter group.",
        required=True,
    )
    argument_parser.add_argument(
        "--max_lr",
        help="""Upper learning rate boundaries in the cycle for each parameter group. Functionally, it defines the
cycle amplitude (max_lr - base_lr). The lr at any cycle is the sum of base_lr and some scaling of
the amplitude; therefore max_lr may not actually be reached depending on scaling function.""",
        required=True,
    )
    argument_parser.add_argument(
        "--step_size_up",
        type=int,
        help="Number of training iterations in the increasing half of a cycle.",
        required=True,
        default=2000,
    )
    argument_parser.add_argument(
        "--step_size_down",
        type=int,
        help="""Number of training iterations in the decreasing half of a cycle. If step_size_down is None, it is
set to step_size_up.""",
    )
    argument_parser.add_argument(
        "--mode",
        help="""One of {triangular, triangular2, exp_range}. Values correspond to policies detailed above. If
scale_fn is not None, this argument is ignored.""",
        required=True,
        default="triangular",
    )
    argument_parser.add_argument("--gamma")
    argument_parser.add_argument("--cycle_momentum")
    argument_parser.add_argument("--scale_fn")
    argument_parser.add_argument("--scale_mode")
    argument_parser.add_argument("--max_momentum")
    argument_parser.add_argument("--last_epoch", required=True, default="triangular")
    argument_parser.add_argument(
        "--base_momentum", type=int, required=True, default=2000
    )
    argument_parser.add_argument("--verbose", type=float, required=True, default=1.0)
    return argument_parser


def ExponentialLRConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :returns: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Decays the learning rate of each parameter group by gamma every epoch.
When last_epoch=-1, sets initial lr as lr."""
    argument_parser.add_argument(
        "--optimizer", help="Wrapped optimizer.", required=True
    )
    argument_parser.add_argument(
        "--gamma",
        type=float,
        help="Multiplicative factor of learning rate decay.",
        required=True,
    )
    argument_parser.add_argument(
        "--last_epoch",
        type=int,
        help="The index of last epoch.",
        required=True,
        default=-1,
    )
    argument_parser.add_argument(
        "--verbose",
        type=bool,
        help="If ``True``, prints a message to stdout for each update.",
        required=True,
        default=False,
    )
    return argument_parser


def LambdaLRConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :returns: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Sets the learning rate of each parameter group to the initial lr
times a given function. When last_epoch=-1, sets initial lr as lr.


Example:
    >>> # Assuming optimizer has two groups.
    >>> lambda1 = lambda epoch: epoch // 30
    >>> lambda2 = lambda epoch: 0.95 ** epoch
    >>> scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
    >>> for epoch in range(100):
    >>>     train(...)
    >>>     validate(...)
    >>>     scheduler.step()"""
    argument_parser.add_argument(
        "--optimizer", help="Wrapped optimizer.", required=True
    )
    argument_parser.add_argument(
        "--lr_lambda",
        help="""A function which computes a multiplicative factor given an integer parameter epoch, or a list of
such functions, one for each group in optimizer.param_groups.""",
        required=True,
    )
    argument_parser.add_argument(
        "--last_epoch",
        type=int,
        help="The index of last epoch.",
        required=True,
        default=-1,
    )
    argument_parser.add_argument(
        "--verbose",
        type=bool,
        help="If ``True``, prints a message to stdout for each update.",
        required=True,
        default=False,
    )
    return argument_parser


def MultiStepLRConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :returns: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Decays the learning rate of each parameter group by gamma once the
number of epoch reaches one of the milestones. Notice that such decay can
happen simultaneously with other changes to the learning rate from outside
this scheduler. When last_epoch=-1, sets initial lr as lr.


Example:
    >>> # Assuming optimizer uses lr = 0.05 for all groups
    >>> # lr = 0.05     if epoch < 30
    >>> # lr = 0.005    if 30 <= epoch < 80
    >>> # lr = 0.0005   if epoch >= 80
    >>> scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
    >>> for epoch in range(100):
    >>>     train(...)
    >>>     validate(...)
    >>>     scheduler.step()"""
    argument_parser.add_argument(
        "--optimizer", help="Wrapped optimizer.", required=True
    )
    argument_parser.add_argument(
        "--milestones", help="List of epoch indices. Must be increasing.", required=True
    )
    argument_parser.add_argument(
        "--gamma",
        type=float,
        help="Multiplicative factor of learning rate decay.",
        required=True,
        default=0.1,
    )
    argument_parser.add_argument(
        "--last_epoch",
        type=int,
        help="The index of last epoch.",
        required=True,
        default=-1,
    )
    argument_parser.add_argument(
        "--verbose",
        type=bool,
        help="If ``True``, prints a message to stdout for each update.",
        required=True,
        default=False,
    )
    return argument_parser


def MultiplicativeLRConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :returns: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Multiply the learning rate of each parameter group by the factor given
in the specified function. When last_epoch=-1, sets initial lr as lr.


Example:
    >>> lmbda = lambda epoch: 0.95
    >>> scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)
    >>> for epoch in range(100):
    >>>     train(...)
    >>>     validate(...)
    >>>     scheduler.step()"""
    argument_parser.add_argument(
        "--optimizer", help="Wrapped optimizer.", required=True
    )
    argument_parser.add_argument(
        "--lr_lambda",
        help="""A function which computes a multiplicative factor given an integer parameter epoch, or a list of
such functions, one for each group in optimizer.param_groups.""",
        required=True,
    )
    argument_parser.add_argument(
        "--last_epoch",
        type=int,
        help="The index of last epoch.",
        required=True,
        default=-1,
    )
    argument_parser.add_argument(
        "--verbose",
        type=bool,
        help="If ``True``, prints a message to stdout for each update.",
        required=True,
        default=False,
    )
    return argument_parser


def OneCycleLRConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :returns: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Sets the learning rate of each parameter group according to the
1cycle learning rate policy. The 1cycle policy anneals the learning
rate from an initial learning rate to some maximum learning rate and then
from that maximum learning rate to some minimum learning rate much lower
than the initial learning rate.
This policy was initially described in the paper `Super-Convergence:
Very Fast Training of Neural Networks Using Large Learning Rates`_.

The 1cycle learning rate policy changes the learning rate after every batch.
`step` should be called after a batch has been used for training.

This scheduler is not chainable.

Note also that the total number of steps in the cycle can be determined in one
of two ways (listed in order of precedence):

#. A value for total_steps is explicitly provided.
#. A number of epochs (epochs) and a number of steps per epoch
   (steps_per_epoch) are provided.
   In this case, the number of total steps is inferred by
   total_steps = epochs * steps_per_epoch

You must either provide a value for total_steps or provide a value for both
epochs and steps_per_epoch.


Example:
    >>> data_loader = torch.utils.data.DataLoader(...)
    >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    >>> scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(data_loader), epochs=10)
    >>> for epoch in range(10):
    >>>     for batch in data_loader:
    >>>         train_batch(...)
    >>>         scheduler.step()


.. _Super-Convergence\\: Very Fast Training of Neural Networks Using Large Learning Rates:
    https://arxiv.org/abs/1708.07120"""
    argument_parser.add_argument(
        "--optimizer", help="Wrapped optimizer.", required=True
    )
    argument_parser.add_argument(
        "--max_lr",
        help="Upper learning rate boundaries in the cycle for each parameter group.",
        required=True,
    )
    argument_parser.add_argument(
        "--total_steps",
        type=int,
        help="""The total number of steps in the cycle. Note that if a value is not provided here, then it must be
inferred by providing a value for epochs and steps_per_epoch.""",
    )
    argument_parser.add_argument(
        "--epochs",
        type=int,
        help="""The number of epochs to train for. This is used along with steps_per_epoch in order to infer the
total number of steps in the cycle if a value for total_steps is not provided.""",
    )
    argument_parser.add_argument(
        "--steps_per_epoch",
        type=int,
        help="""The number of steps per epoch to train for. This is used along with epochs in order to infer the
total number of steps in the cycle if a value for total_steps is not provided.""",
    )
    argument_parser.add_argument(
        "--pct_start",
        type=float,
        help="The percentage of the cycle (in number of steps) spent increasing the learning rate.",
        required=True,
        default=0.3,
    )
    argument_parser.add_argument(
        "--anneal_strategy",
        choices=("cos", "linear"),
        help='Specifies the annealing strategy: "cos" for cosine annealing, "linear" for linear annealing.',
        required=True,
        default="cos",
    )
    argument_parser.add_argument(
        "--cycle_momentum",
        type=bool,
        help="""If ``True``, momentum is cycled inversely to learning rate between 'base_momentum' and
'max_momentum'.""",
        required=True,
        default=True,
    )
    argument_parser.add_argument(
        "--base_momentum",
        type=float,
        help="""Lower momentum boundaries in the cycle for each parameter group. Note that momentum is cycled
inversely to learning rate; at the peak of a cycle, momentum is 'base_momentum' and learning rate is
'max_lr'.""",
        required=True,
        default=0.85,
    )
    argument_parser.add_argument(
        "--max_momentum",
        type=float,
        help="""Upper momentum boundaries in the cycle for each parameter group. Functionally, it defines the cycle
amplitude (max_momentum - base_momentum). Note that momentum is cycled inversely to learning rate;
at the start of a cycle, momentum is 'max_momentum' and learning rate is 'base_lr'""",
        required=True,
        default=0.95,
    )
    argument_parser.add_argument(
        "--div_factor",
        type=float,
        help="Determines the initial learning rate via initial_lr = max_lr/div_factor",
        required=True,
        default=25.0,
    )
    argument_parser.add_argument(
        "--final_div_factor",
        type=float,
        help="Determines the minimum learning rate via min_lr = initial_lr/final_div_factor",
        required=True,
        default=10000.0,
    )
    argument_parser.add_argument(
        "--last_epoch",
        type=int,
        help="""The index of the last batch. This parameter is used when resuming a training job. Since `step()`
should be invoked after each batch instead of after each epoch, this number represents the total
number of *batches* computed, not the total number of epochs computed. When last_epoch=-1, the
schedule is started from the beginning.""",
        required=True,
        default=-1,
    )
    argument_parser.add_argument(
        "--verbose",
        type=bool,
        help="If ``True``, prints a message to stdout for each update.",
        required=True,
        default=False,
    )
    return argument_parser


def ReduceLROnPlateauConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :returns: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Reduce learning rate when a metric has stopped improving.
Models often benefit from reducing the learning rate by a factor
of 2-10 once learning stagnates. This scheduler reads a metrics
quantity and if no improvement is seen for a 'patience' number
of epochs, the learning rate is reduced.


Example:
    >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    >>> scheduler = ReduceLROnPlateau(optimizer, 'min')
    >>> for epoch in range(10):
    >>>     train(...)
    >>>     val_loss = validate(...)
    >>>     # Note that step should be called after validate()
    >>>     scheduler.step(val_loss)"""
    argument_parser.add_argument(
        "--optimizer", help="Wrapped optimizer.", required=True
    )
    argument_parser.add_argument(
        "--mode",
        help="""One of `min`, `max`. In `min` mode, lr will be reduced when the quantity monitored has stopped
decreasing; in `max` mode it will be reduced when the quantity monitored has stopped increasing.""",
        required=True,
        default="min",
    )
    argument_parser.add_argument(
        "--factor",
        type=float,
        help="Factor by which the learning rate will be reduced. new_lr = lr * factor.",
        required=True,
        default=0.1,
    )
    argument_parser.add_argument(
        "--patience",
        type=int,
        help="""Number of epochs with no improvement after which learning rate will be reduced. For example, if
`patience = 2`, then we will ignore the first 2 epochs with no improvement, and will only decrease
the LR after the 3rd epoch if the loss still hasn't improved then.""",
        required=True,
        default=10,
    )
    argument_parser.add_argument(
        "--threshold",
        type=float,
        help="Threshold for measuring the new optimum, to only focus on significant changes.",
        required=True,
        default=0.0001,
    )
    argument_parser.add_argument(
        "--threshold_mode",
        help="""One of `rel`, `abs`. In `rel` mode, dynamic_threshold = best * ( 1 + threshold ) in 'max' mode or
best * ( 1 - threshold ) in `min` mode. In `abs` mode, dynamic_threshold = best + threshold in `max`
mode or best - threshold in `min` mode.""",
        required=True,
        default="rel",
    )
    argument_parser.add_argument(
        "--cooldown",
        type=int,
        help="Number of epochs to wait before resuming normal operation after lr has been reduced.",
        required=True,
        default=0,
    )
    argument_parser.add_argument(
        "--min_lr",
        type=int,
        help="""A scalar or a list of scalars. A lower bound on the learning rate of all param groups or each group
respectively.""",
        required=True,
        default=0,
    )
    argument_parser.add_argument(
        "--eps",
        type=float,
        help="""Minimal decay applied to lr. If the difference between new and old lr is smaller than eps, the
update is ignored.""",
        required=True,
        default=1e-08,
    )
    argument_parser.add_argument(
        "--verbose",
        type=bool,
        help="If ``True``, prints a message to stdout for each update.",
        required=True,
        default=False,
    )
    return argument_parser


def StepLRConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :returns: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = """Decays the learning rate of each parameter group by gamma every
step_size epochs. Notice that such decay can happen simultaneously with
other changes to the learning rate from outside this scheduler. When
last_epoch=-1, sets initial lr as lr.


Example:
    >>> # Assuming optimizer uses lr = 0.05 for all groups
    >>> # lr = 0.05     if epoch < 30
    >>> # lr = 0.005    if 30 <= epoch < 60
    >>> # lr = 0.0005   if 60 <= epoch < 90
    >>> # ...
    >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    >>> for epoch in range(100):
    >>>     train(...)
    >>>     validate(...)
    >>>     scheduler.step()"""
    argument_parser.add_argument(
        "--optimizer", help="Wrapped optimizer.", required=True
    )
    argument_parser.add_argument(
        "--step_size", type=int, help="Period of learning rate decay.", required=True
    )
    argument_parser.add_argument(
        "--gamma",
        type=float,
        help="Multiplicative factor of learning rate decay.",
        required=True,
        default=0.1,
    )
    argument_parser.add_argument(
        "--last_epoch",
        type=int,
        help="The index of last epoch.",
        required=True,
        default=-1,
    )
    argument_parser.add_argument(
        "--verbose",
        type=bool,
        help="If ``True``, prints a message to stdout for each update.",
        required=True,
        default=False,
    )
    return argument_parser


__all__ = [
    "CosineAnnealingLRConfig",
    "CosineAnnealingWarmRestartsConfig",
    "CounterConfig",
    "CyclicLRConfig",
    "ExponentialLRConfig",
    "LambdaLRConfig",
    "MultiStepLRConfig",
    "MultiplicativeLRConfig",
    "OneCycleLRConfig",
    "ReduceLROnPlateauConfig",
    "StepLRConfig",
]
