""" Generated Optimizer LR scheduler CLI parsers """
from yaml import safe_load as loads


def CosineAnnealingLRConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Set the learning rate of each parameter group using a cosine annealing\nschedule, where :math:`\\eta_{max}` is set to the initial lr and\n:math:`T_{cur}` is the number of epochs since the last restart in SGDR:\n\n.. math::\n    \\begin{aligned}\n        \\eta_t & = \\eta_{min} + \\frac{1}{2}(\\eta_{max} - \\eta_{min})\\left(1\n        + \\cos\\left(\\frac{T_{cur}}{T_{max}}\\pi\\right)\\right),\n        & T_{cur} \\neq (2k+1)T_{max}; \\\\\n        \\eta_{t+1} & = \\eta_{t} + \\frac{1}{2}(\\eta_{max} - \\eta_{min})\n        \\left(1 - \\cos\\left(\\frac{1}{T_{max}}\\pi\\right)\\right),\n        & T_{cur} = (2k+1)T_{max}.\n    \\end{aligned}\n\nWhen last_epoch=-1, sets initial lr as lr. Notice that because the schedule\nis defined recursively, the learning rate can be simultaneously modified\noutside this scheduler by other operators. If the learning rate is set\nsolely by this scheduler, the learning rate at each step becomes:\n\n.. math::\n    \\eta_t = \\eta_{min} + \\frac{1}{2}(\\eta_{max} - \\eta_{min})\\left(1 +\n    \\cos\\left(\\frac{T_{cur}}{T_{max}}\\pi\\right)\\right)\n\nIt has been proposed in\n`SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only\nimplements the cosine annealing part of SGDR, and not the restarts.\n\n\n.. _SGDR\\: Stochastic Gradient Descent with Warm Restarts:\n    https://arxiv.org/abs/1608.03983"
    argument_parser.add_argument(
        "--optimizer", type=int, help="Wrapped optimizer.", required=True, default=0
    )
    argument_parser.add_argument(
        "--T_max",
        type=int,
        help="Maximum number of iterations.",
        required=True,
        default="```-1```",
    )
    argument_parser.add_argument(
        "--eta_min", type=float, help="Minimum learning rate.", required=True, default=0
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
        help="If ``True``, prints a message to stdout for\n        each update.",
        required=True,
        default=False,
    )
    return argument_parser


def CosineAnnealingWarmRestartsConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Set the learning rate of each parameter group using a cosine annealing\nschedule, where :math:`\\eta_{max}` is set to the initial lr, :math:`T_{cur}`\nis the number of epochs since the last restart and :math:`T_{i}` is the number\nof epochs between two warm restarts in SGDR:\n\n.. math::\n    \\eta_t = \\eta_{min} + \\frac{1}{2}(\\eta_{max} - \\eta_{min})\\left(1 +\n    \\cos\\left(\\frac{T_{cur}}{T_{i}}\\pi\\right)\\right)\n\nWhen :math:`T_{cur}=T_{i}`, set :math:`\\eta_t = \\eta_{min}`.\nWhen :math:`T_{cur}=0` after restart, set :math:`\\eta_t=\\eta_{max}`.\n\nIt has been proposed in\n`SGDR: Stochastic Gradient Descent with Warm Restarts`_.\n\n\n.. _SGDR\\: Stochastic Gradient Descent with Warm Restarts:\n    https://arxiv.org/abs/1608.03983"
    argument_parser.add_argument(
        "--optimizer", type=int, help="Wrapped optimizer.", required=True, default=1
    )
    argument_parser.add_argument(
        "--T_0",
        type=int,
        help="Number of iterations for the first restart.",
        required=True,
        default=0,
    )
    argument_parser.add_argument(
        "--T_mult",
        type=int,
        help="A factor increases :math:`T_{i}` after a restart.",
        default=1,
    )
    argument_parser.add_argument(
        "--eta_min", type=float, help="Minimum learning rate.", default=0
    )
    argument_parser.add_argument(
        "--last_epoch", type=int, help="The index of last epoch.", default=-1
    )
    argument_parser.add_argument(
        "--verbose",
        type=bool,
        help="If ``True``, prints a message to stdout for\n        each update.",
        required=True,
        default=False,
    )
    return argument_parser


def CounterConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Dict subclass for counting hashable items.  Sometimes called a bag\nor multiset.  Elements are stored as dictionary keys and their counts\nare stored as dictionary values.\n\n>>> c = Counter('abcdeabcdabcaba')  # count elements from a string\n\n>>> c.most_common(3)                # three most common elements\n[('a', 5), ('b', 4), ('c', 3)]\n>>> sorted(c)                       # list all unique elements\n['a', 'b', 'c', 'd', 'e']\n>>> ''.join(sorted(c.elements()))   # list elements with repetitions\n'aaaaabbbbcccdde'\n>>> sum(c.values())                 # total of all counts\n15\n\n>>> c['a']                          # count of letter 'a'\n5\n>>> for elem in 'shazam':           # update counts from an iterable\n...     c[elem] += 1                # by adding 1 to each element's count\n>>> c['a']                          # now there are seven 'a'\n7\n>>> del c['b']                      # remove all 'b'\n>>> c['b']                          # now there are zero 'b'\n0\n\n>>> d = Counter('simsalabim')       # make another counter\n>>> c.update(d)                     # add in the second counter\n>>> c['a']                          # now there are nine 'a'\n9\n\n>>> c.clear()                       # empty the counter\n>>> c\nCounter()\n\nNote:  If a count is set to zero or reduced to zero, it will remain\nin the counter until the entry is deleted or the counter is cleared:\n\n>>> c = Counter('aaabbc')\n>>> c['b'] -= 2                     # reduce the count of 'b' by two\n>>> c.most_common()                 # 'b' is still in, but its count is zero\n[('a', 3), ('c', 1), ('b', 0)]"
    argument_parser.add_argument("--kwds", type=loads, help="", required=True)
    return argument_parser


def CyclicLRConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Sets the learning rate of each parameter group according to\ncyclical learning rate policy (CLR). The policy cycles the learning\nrate between two boundaries with a constant frequency, as detailed in\nthe paper `Cyclical Learning Rates for Training Neural Networks`_.\nThe distance between the two boundaries can be scaled on a per-iteration\nor per-cycle basis.\n\nCyclical learning rate policy changes the learning rate after every batch.\n`step` should be called after a batch has been used for training.\n\nThis class has three built-in policies, as put forth in the paper:\n\n* \"triangular\": A basic triangular cycle without amplitude scaling.\n* \"triangular2\": A basic triangular cycle that scales initial amplitude by half each cycle.\n* \"exp_range\": A cycle that scales initial amplitude by :math:`\\text{gamma}^{\\text{cycle iterations}}`\n  at each cycle iteration.\n\nThis implementation was adapted from the github repo: `bckenstler/CLR`_\n\n\n    gamma (float): Constant in 'exp_range' scaling function:\n            gamma**(cycle iterations)\n            Default: 1.0\n        scale_fn (function): Custom scaling policy defined by a single\n            argument lambda function, where\n            0 <= scale_fn(x) <= 1 for all x >= 0.\n            If specified, then 'mode' is ignored.\n            Default: None\n        scale_mode (str): {'cycle', 'iterations'}.\n            Defines whether scale_fn is evaluated on\n            cycle number or cycle iterations (training\n            iterations since start of cycle).\n            Default: 'cycle'\n        cycle_momentum (bool): If ``True``, momentum is cycled inversely\n            to learning rate between 'base_momentum' and 'max_momentum'.\n            Default: True\n        base_momentum (float or list): Lower momentum boundaries in the cycle\n            for each parameter group. Note that momentum is cycled inversely\n            to learning rate; at the peak of a cycle, momentum is\n            'base_momentum' and learning rate is 'max_lr'.\n            Default: 0.8\n        max_momentum (float or list): Upper momentum boundaries in the cycle\n            for each parameter group. Functionally,\n            it defines the cycle amplitude (max_momentum - base_momentum).\n            The momentum at any cycle is the difference of max_momentum\n            and some scaling of the amplitude; therefore\n            base_momentum may not actually be reached depending on\n            scaling function. Note that momentum is cycled inversely\n            to learning rate; at the start of a cycle, momentum is 'max_momentum'\n            and learning rate is 'base_lr'\n            Default: 0.9\n        last_epoch (int): The index of the last batch. This parameter is used when\n            resuming a training job. Since `step()` should be invoked after each\n            batch instead of after each epoch, this number represents the total\n            number of *batches* computed, not the total number of epochs computed.\n            When last_epoch=-1, the schedule is started from the beginning.\n            Default: -1\n        verbose (bool): If ``True``, prints a message to stdout for\n            each update. Default: ``False``.\n\n\nExample:\n    >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)\n    >>> scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)\n    >>> data_loader = torch.utils.data.DataLoader(...)\n    >>> for epoch in range(10):\n    >>>     for batch in data_loader:\n    >>>         train_batch(...)\n    >>>         scheduler.step()\n\n\n.. _Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186\n.. _bckenstler/CLR: https://github.com/bckenstler/CLR"
    argument_parser.add_argument(
        "--optimizer", type=int, help="Wrapped optimizer.", required=True, default=2000
    )
    argument_parser.add_argument(
        "--base_lr",
        type=str,
        help="Initial learning rate which is the\n        lower boundary in the cycle for each parameter group.",
    )
    argument_parser.add_argument(
        "--max_lr",
        type=str,
        help="Upper learning rate boundaries in the cycle\n        for each parameter group. Functionally,\n        it defines the cycle amplitude (max_lr - base_lr).\n        The lr at any cycle is the sum of base_lr\n        and some scaling of the amplitude; therefore\n        max_lr may not actually be reached depending on\n        scaling function.",
        required=True,
        default="triangular",
    )
    argument_parser.add_argument(
        "--step_size_up",
        type=int,
        help="Number of training iterations in the\n        increasing half of a cycle.",
        required=True,
        default=2000,
    )
    argument_parser.add_argument(
        "--step_size_down",
        type=int,
        help="Number of training iterations in the\n        decreasing half of a cycle. If step_size_down is None,\n        it is set to step_size_up.",
        required=True,
        default="None",
    )
    argument_parser.add_argument(
        "--mode",
        type=str,
        help="One of {triangular, triangular2, exp_range}.\n        Values correspond to policies detailed above.\n        If scale_fn is not None, this argument is ignored.\n       ",
        required=True,
        default="triangular",
    )
    argument_parser.add_argument("--last_epoch", help=None, required=True)
    argument_parser.add_argument("--gamma", help=None, required=True, default=True)
    argument_parser.add_argument("--verbose", help=None, required=True)
    argument_parser.add_argument(
        "--cycle_momentum", help=None, required=True, default="```-1```"
    )
    argument_parser.add_argument("--scale_mode", help=None, required=True, default=0.9)
    argument_parser.add_argument(
        "--base_momentum", help=None, required=True, default=False
    )
    argument_parser.add_argument("--scale_fn", help=None, required=True, default=0.8)
    argument_parser.add_argument("--max_momentum", help=None, required=True)
    return argument_parser


def ExponentialLRConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Decays the learning rate of each parameter group by gamma every epoch.\nWhen last_epoch=-1, sets initial lr as lr."
    argument_parser.add_argument(
        "--optimizer",
        type=str,
        help="Wrapped optimizer.",
        required=True,
        default="```-1```",
    )
    argument_parser.add_argument(
        "--gamma",
        type=float,
        help="Multiplicative factor of learning rate decay.",
        required=True,
        default=False,
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
        help="If ``True``, prints a message to stdout for\n        each update.",
        required=True,
        default=False,
    )
    return argument_parser


def LambdaLRConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Sets the learning rate of each parameter group to the initial lr\ntimes a given function. When last_epoch=-1, sets initial lr as lr.\n\n\nExample:\n    >>> # Assuming optimizer has two groups.\n    >>> lambda1 = lambda epoch: epoch // 30\n    >>> lambda2 = lambda epoch: 0.95 ** epoch\n    >>> scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])\n    >>> for epoch in range(100):\n    >>>     train(...)\n    >>>     validate(...)\n    >>>     scheduler.step()"
    argument_parser.add_argument(
        "--optimizer",
        type=str,
        help="Wrapped optimizer.",
        required=True,
        default="```-1```",
    )
    argument_parser.add_argument(
        "--lr_lambda",
        type=bool,
        help="A function which computes a multiplicative\n        factor given an integer parameter epoch, or a list of such\n        functions, one for each group in optimizer.param_groups.",
        required=True,
        default=False,
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
        help="If ``True``, prints a message to stdout for\n        each update.",
        required=True,
        default=False,
    )
    return argument_parser


def MultiStepLRConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Decays the learning rate of each parameter group by gamma once the\nnumber of epoch reaches one of the milestones. Notice that such decay can\nhappen simultaneously with other changes to the learning rate from outside\nthis scheduler. When last_epoch=-1, sets initial lr as lr.\n\n\nExample:\n    >>> # Assuming optimizer uses lr = 0.05 for all groups\n    >>> # lr = 0.05     if epoch < 30\n    >>> # lr = 0.005    if 30 <= epoch < 80\n    >>> # lr = 0.0005   if epoch >= 80\n    >>> scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)\n    >>> for epoch in range(100):\n    >>>     train(...)\n    >>>     validate(...)\n    >>>     scheduler.step()"
    argument_parser.add_argument(
        "--optimizer", type=float, help="Wrapped optimizer.", required=True, default=0.1
    )
    argument_parser.add_argument(
        "--milestones",
        type=str,
        help="List of epoch indices. Must be increasing.",
        required=True,
        default="```-1```",
    )
    argument_parser.add_argument(
        "--gamma",
        type=float,
        help="Multiplicative factor of learning rate decay.\n       ",
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
        help="If ``True``, prints a message to stdout for\n        each update.",
        required=True,
        default=False,
    )
    return argument_parser


def MultiplicativeLRConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Multiply the learning rate of each parameter group by the factor given\nin the specified function. When last_epoch=-1, sets initial lr as lr.\n\n\nExample:\n    >>> lmbda = lambda epoch: 0.95\n    >>> scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)\n    >>> for epoch in range(100):\n    >>>     train(...)\n    >>>     validate(...)\n    >>>     scheduler.step()"
    argument_parser.add_argument(
        "--optimizer",
        type=str,
        help="Wrapped optimizer.",
        required=True,
        default="```-1```",
    )
    argument_parser.add_argument(
        "--lr_lambda",
        type=bool,
        help="A function which computes a multiplicative\n        factor given an integer parameter epoch, or a list of such\n        functions, one for each group in optimizer.param_groups.",
        required=True,
        default=False,
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
        help="If ``True``, prints a message to stdout for\n        each update.",
        required=True,
        default=False,
    )
    return argument_parser


def OneCycleLRConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Sets the learning rate of each parameter group according to the\n1cycle learning rate policy. The 1cycle policy anneals the learning\nrate from an initial learning rate to some maximum learning rate and then\nfrom that maximum learning rate to some minimum learning rate much lower\nthan the initial learning rate.\nThis policy was initially described in the paper `Super-Convergence:\nVery Fast Training of Neural Networks Using Large Learning Rates`_.\n\nThe 1cycle learning rate policy changes the learning rate after every batch.\n`step` should be called after a batch has been used for training.\n\nThis scheduler is not chainable.\n\nNote also that the total number of steps in the cycle can be determined in one\nof two ways (listed in order of precedence):\n\n#. A value for total_steps is explicitly provided.\n#. A number of epochs (epochs) and a number of steps per epoch\n   (steps_per_epoch) are provided.\n   In this case, the number of total steps is inferred by\n   total_steps = epochs * steps_per_epoch\n\nYou must either provide a value for total_steps or provide a value for both\nepochs and steps_per_epoch.\n\n\nExample:\n    >>> data_loader = torch.utils.data.DataLoader(...)\n    >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)\n    >>> scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(data_loader), epochs=10)\n    >>> for epoch in range(10):\n    >>>     for batch in data_loader:\n    >>>         train_batch(...)\n    >>>         scheduler.step()\n\n\n.. _Super-Convergence\\: Very Fast Training of Neural Networks Using Large Learning Rates:\n    https://arxiv.org/abs/1708.07120"
    argument_parser.add_argument("--optimizer", type=str, help="Wrapped optimizer.")
    argument_parser.add_argument(
        "--max_lr",
        type=str,
        help="Upper learning rate boundaries in the cycle\n        for each parameter group.",
    )
    argument_parser.add_argument(
        "--total_steps",
        type=int,
        help="The total number of steps in the cycle. Note that\n        if a value is not provided here, then it must be inferred by providing\n        a value for epochs and steps_per_epoch.\n       ",
        required=True,
        default="None",
    )
    argument_parser.add_argument(
        "--epochs",
        type=int,
        help="The number of epochs to train for. This is used along\n        with steps_per_epoch in order to infer the total number of steps in the cycle\n        if a value for total_steps is not provided.\n       ",
        required=True,
        default="None",
    )
    argument_parser.add_argument(
        "--steps_per_epoch",
        type=int,
        help="The number of steps per epoch to train for. This is\n        used along with epochs in order to infer the total number of steps in the\n        cycle if a value for total_steps is not provided.\n       ",
        required=True,
        default="None",
    )
    argument_parser.add_argument(
        "--pct_start",
        type=float,
        help="The percentage of the cycle (in number of steps) spent\n        increasing the learning rate.\n       ",
        required=True,
        default=0.3,
    )
    argument_parser.add_argument(
        "--anneal_strategy",
        type=str,
        help="{'cos', 'linear'}\n        Specifies the annealing strategy: \"cos\" for cosine annealing, \"linear\" for\n        linear annealing.\n       ",
        required=True,
        default="cos",
    )
    argument_parser.add_argument(
        "--cycle_momentum",
        type=bool,
        help="If ``True``, momentum is cycled inversely\n        to learning rate between 'base_momentum' and 'max_momentum'.\n       ",
        required=True,
        default=True,
    )
    argument_parser.add_argument(
        "--base_momentum",
        type=float,
        help="Lower momentum boundaries in the cycle\n        for each parameter group. Note that momentum is cycled inversely\n        to learning rate; at the peak of a cycle, momentum is\n        'base_momentum' and learning rate is 'max_lr'.\n       ",
        required=True,
        default=0.85,
    )
    argument_parser.add_argument(
        "--max_momentum",
        type=float,
        help="Upper momentum boundaries in the cycle\n        for each parameter group. Functionally,\n        it defines the cycle amplitude (max_momentum - base_momentum).\n        Note that momentum is cycled inversely\n        to learning rate; at the start of a cycle, momentum is 'max_momentum'\n        and learning rate is 'base_lr'\n       ",
        required=True,
        default=0.95,
    )
    argument_parser.add_argument(
        "--div_factor",
        type=float,
        help="Determines the initial learning rate via\n        initial_lr = max_lr/div_factor\n       ",
        required=True,
        default=25.0,
    )
    argument_parser.add_argument(
        "--final_div_factor",
        type=float,
        help="Determines the minimum learning rate via\n        min_lr = initial_lr/final_div_factor\n       ",
        required=True,
        default=10000.0,
    )
    argument_parser.add_argument(
        "--last_epoch",
        type=int,
        help="The index of the last batch. This parameter is used when\n        resuming a training job. Since `step()` should be invoked after each\n        batch instead of after each epoch, this number represents the total\n        number of *batches* computed, not the total number of epochs computed.\n        When last_epoch=-1, the schedule is started from the beginning.\n       ",
        required=True,
        default=-1,
    )
    argument_parser.add_argument(
        "--verbose",
        type=bool,
        help="If ``True``, prints a message to stdout for\n        each update.",
        required=True,
        default=False,
    )
    return argument_parser


def ReduceLROnPlateauConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Reduce learning rate when a metric has stopped improving.\nModels often benefit from reducing the learning rate by a factor\nof 2-10 once learning stagnates. This scheduler reads a metrics\nquantity and if no improvement is seen for a 'patience' number\nof epochs, the learning rate is reduced.\n\n\nExample:\n    >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)\n    >>> scheduler = ReduceLROnPlateau(optimizer, 'min')\n    >>> for epoch in range(10):\n    >>>     train(...)\n    >>>     val_loss = validate(...)\n    >>>     # Note that step should be called after validate()\n    >>>     scheduler.step(val_loss)"
    argument_parser.add_argument(
        "--optimizer", type=str, help="Wrapped optimizer.", required=True, default="min"
    )
    argument_parser.add_argument(
        "--mode",
        type=str,
        help="One of `min`, `max`. In `min` mode, lr will\n        be reduced when the quantity monitored has stopped\n        decreasing; in `max` mode it will be reduced when the\n        quantity monitored has stopped increasing.",
        required=True,
        default="min",
    )
    argument_parser.add_argument(
        "--factor",
        type=float,
        help="Factor by which the learning rate will be\n        reduced. new_lr = lr * factor.",
        required=True,
        default=0.1,
    )
    argument_parser.add_argument(
        "--patience",
        type=int,
        help="Number of epochs with no improvement after\n        which learning rate will be reduced. For example, if\n        `patience = 2`, then we will ignore the first 2 epochs\n        with no improvement, and will only decrease the LR after the\n        3rd epoch if the loss still hasn't improved then.\n       ",
        required=True,
        default=10,
    )
    argument_parser.add_argument(
        "--threshold",
        type=float,
        help="Threshold for measuring the new optimum,\n        to only focus on significant changes.",
        required=True,
        default=0.0001,
    )
    argument_parser.add_argument(
        "--threshold_mode",
        type=str,
        help="One of `rel`, `abs`. In `rel` mode,\n        dynamic_threshold = best * ( 1 + threshold ) in 'max'\n        mode or best * ( 1 - threshold ) in `min` mode.\n        In `abs` mode, dynamic_threshold = best + threshold in\n        `max` mode or best - threshold in `min` mode.",
        required=True,
        default="rel",
    )
    argument_parser.add_argument(
        "--cooldown",
        type=int,
        help="Number of epochs to wait before resuming\n        normal operation after lr has been reduced.",
        required=True,
        default=0,
    )
    argument_parser.add_argument(
        "--min_lr",
        type=int,
        help="A scalar or a list of scalars. A\n        lower bound on the learning rate of all param groups\n        or each group respectively.",
        required=True,
        default=0,
    )
    argument_parser.add_argument(
        "--eps",
        type=float,
        help="Minimal decay applied to lr. If the difference\n        between new and old lr is smaller than eps, the update is\n        ignored.",
        required=True,
        default=1e-08,
    )
    argument_parser.add_argument(
        "--verbose",
        type=bool,
        help="If ``True``, prints a message to stdout for\n        each update.",
        required=True,
        default=False,
    )
    return argument_parser


def StepLRConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Decays the learning rate of each parameter group by gamma every\nstep_size epochs. Notice that such decay can happen simultaneously with\nother changes to the learning rate from outside this scheduler. When\nlast_epoch=-1, sets initial lr as lr.\n\n\nExample:\n    >>> # Assuming optimizer uses lr = 0.05 for all groups\n    >>> # lr = 0.05     if epoch < 30\n    >>> # lr = 0.005    if 30 <= epoch < 60\n    >>> # lr = 0.0005   if 60 <= epoch < 90\n    >>> # ...\n    >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)\n    >>> for epoch in range(100):\n    >>>     train(...)\n    >>>     validate(...)\n    >>>     scheduler.step()"
    argument_parser.add_argument(
        "--optimizer", type=float, help="Wrapped optimizer.", required=True, default=0.1
    )
    argument_parser.add_argument(
        "--step_size",
        type=int,
        help="Period of learning rate decay.",
        required=True,
        default="```-1```",
    )
    argument_parser.add_argument(
        "--gamma",
        type=float,
        help="Multiplicative factor of learning rate decay.\n       ",
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
        help="If ``True``, prints a message to stdout for\n        each update.",
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
