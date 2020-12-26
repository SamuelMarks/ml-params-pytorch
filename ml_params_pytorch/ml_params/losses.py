""" Generated Losses CLI parsers """

from yaml import safe_load as loads


def BCELossConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Creates a criterion that measures the Binary Cross Entropy\nbetween the target and the output:\n\nThe unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:\n\n.. math::\n    \\ell(x, y) = L = \\{l_1,\\dots,l_N\\}^\\top, \\quad\n    l_n = - w_n \\left[ y_n \\cdot \\log x_n + (1 - y_n) \\cdot \\log (1 - x_n) \\right],\n\nwhere :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``\n(default ``'mean'``), then\n\n.. math::\n    \\ell(x, y) = \\begin{cases}\n        \\operatorname{mean}(L), & \\text{if reduction} = \\text{'mean';}\\\\\n        \\operatorname{sum}(L),  & \\text{if reduction} = \\text{'sum'.}\n    \\end{cases}\n\nThis is used for measuring the error of a reconstruction in for example\nan auto-encoder. Note that the targets :math:`y` should be numbers\nbetween 0 and 1.\n\nNotice that if :math:`x_n` is either 0 or 1, one of the log terms would be\nmathematically undefined in the above loss equation. PyTorch chooses to set\n:math:`\\log (0) = -\\infty`, since :math:`\\lim_{x\\to 0} \\log (x) = -\\infty`.\nHowever, an infinite term in the loss equation is not desirable for several reasons.\n\nFor one, if either :math:`y_n = 0` or :math:`(1 - y_n) = 0`, then we would be\nmultiplying 0 with infinity. Secondly, if we have an infinite loss value, then\nwe would also have an infinite term in our gradient, since\n:math:`\\lim_{x\\to 0} \\frac{d}{dx} \\log (x) = \\infty`.\nThis would make BCELoss's backward method nonlinear with respect to :math:`x_n`,\nand using it for things like linear regression would not be straight-forward.\n\nOur solution is that BCELoss clamps its log function outputs to be greater than\nor equal to -100. This way, we can always have a finite loss value and a linear\nbackward method.\n\n\n    reduction (string, optional): Specifies the reduction to apply to the output:\n            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,\n            ``'mean'``: the sum of the output will be divided by the number of\n            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`\n            and :attr:`reduce` are in the process of being deprecated, and in the meantime,\n            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``\n\n\nShape:\n    - Input: :math:`(N, *)` where :math:`*` means, any number of additional\n      dimensions\n    - Target: :math:`(N, *)`, same shape as the input\n    - Output: scalar. If :attr:`reduction` is ``'none'``, then :math:`(N, *)`, same\n      shape as input.\n\nExamples::\n\n    >>> m = nn.Sigmoid()\n    >>> loss = nn.BCELoss()\n    >>> input = torch.randn(3, requires_grad=True)\n    >>> target = torch.empty(3).random_(2)\n    >>> output = loss(m(input), target)\n    >>> output.backward()"
    argument_parser.add_argument(
        "--weight",
        type=str,
        help="a manual rescaling weight given to the loss\n        of each batch element. If given, has to be a Tensor of size `nbatch`.",
    )
    argument_parser.add_argument(
        "--size_average",
        type=bool,
        help="Deprecated (see :attr:`reduction`). By default,\n        the losses are averaged over each loss element in the batch. Note that for\n        some losses, there are multiple elements per sample. If the field :attr:`size_average`\n        is set to ``False``, the losses are instead summed for each minibatch. Ignored\n        when reduce is ``False``.",
        default=True,
    )
    argument_parser.add_argument(
        "--reduce",
        type=bool,
        help="Deprecated (see :attr:`reduction`). By default, the\n        losses are averaged or summed over observations for each minibatch depending\n        on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per\n        batch element instead and ignores :attr:`size_average`.",
        default=True,
    )
    argument_parser.add_argument(
        "--__constants__",
        type=str,
        action="append",
        help="",
        required=True,
        default="reduction",
    )
    argument_parser.add_argument(
        "--reduction", type=str, help=None, required=True, default="mean"
    )
    return argument_parser


def BCEWithLogitsLossConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "This loss combines a `Sigmoid` layer and the `BCELoss` in one single\nclass. This version is more numerically stable than using a plain `Sigmoid`\nfollowed by a `BCELoss` as, by combining the operations into one layer,\nwe take advantage of the log-sum-exp trick for numerical stability.\n\nThe unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:\n\n.. math::\n    \\ell(x, y) = L = \\{l_1,\\dots,l_N\\}^\\top, \\quad\n    l_n = - w_n \\left[ y_n \\cdot \\log \\sigma(x_n)\n    + (1 - y_n) \\cdot \\log (1 - \\sigma(x_n)) \\right],\n\nwhere :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``\n(default ``'mean'``), then\n\n.. math::\n    \\ell(x, y) = \\begin{cases}\n        \\operatorname{mean}(L), & \\text{if reduction} = \\text{'mean';}\\\\\n        \\operatorname{sum}(L),  & \\text{if reduction} = \\text{'sum'.}\n    \\end{cases}\n\nThis is used for measuring the error of a reconstruction in for example\nan auto-encoder. Note that the targets `t[i]` should be numbers\nbetween 0 and 1.\n\nIt's possible to trade off recall and precision by adding weights to positive examples.\nIn the case of multi-label classification the loss can be described as:\n\n.. math::\n    \\ell_c(x, y) = L_c = \\{l_{1,c},\\dots,l_{N,c}\\}^\\top, \\quad\n    l_{n,c} = - w_{n,c} \\left[ p_c y_{n,c} \\cdot \\log \\sigma(x_{n,c})\n    + (1 - y_{n,c}) \\cdot \\log (1 - \\sigma(x_{n,c})) \\right],\n\nwhere :math:`c` is the class number (:math:`c > 1` for multi-label binary classification,\n:math:`c = 1` for single-label binary classification),\n:math:`n` is the number of the sample in the batch and\n:math:`p_c` is the weight of the positive answer for the class :math:`c`.\n\n:math:`p_c > 1` increases the recall, :math:`p_c < 1` increases the precision.\n\nFor example, if a dataset contains 100 positive and 300 negative examples of a single class,\nthen `pos_weight` for the class should be equal to :math:`\\frac{300}{100}=3`.\nThe loss would act as if the dataset contains :math:`3\\times 100=300` positive examples.\n\nExamples::\n\n    >>> target = torch.ones([10, 64], dtype=torch.float32)  # 64 classes, batch size = 10\n    >>> output = torch.full([10, 64], 1.5)  # A prediction (logit)\n    >>> pos_weight = torch.ones([64])  # All weights are equal to 1\n    >>> criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)\n    >>> criterion(output, target)  # -log(sigmoid(1.5))\n    tensor(0.2014)\n\n\n    reduction (string, optional): Specifies the reduction to apply to the output:\n            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,\n            ``'mean'``: the sum of the output will be divided by the number of\n            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`\n            and :attr:`reduce` are in the process of being deprecated, and in the meantime,\n            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``\n        pos_weight (Tensor, optional): a weight of positive examples.\n                Must be a vector with length equal to the number of classes.\n\n\nShape:\n    - Input: :math:`(N, *)` where :math:`*` means, any number of additional dimensions\n    - Target: :math:`(N, *)`, same shape as the input\n    - Output: scalar. If :attr:`reduction` is ``'none'``, then :math:`(N, *)`, same\n      shape as input.\n\n Examples::\n\n    >>> loss = nn.BCEWithLogitsLoss()\n    >>> input = torch.randn(3, requires_grad=True)\n    >>> target = torch.empty(3).random_(2)\n    >>> output = loss(input, target)\n    >>> output.backward()"
    argument_parser.add_argument(
        "--weight",
        type=str,
        help="a manual rescaling weight given to the loss\n        of each batch element. If given, has to be a Tensor of size `nbatch`.",
    )
    argument_parser.add_argument(
        "--size_average",
        type=bool,
        help="Deprecated (see :attr:`reduction`). By default,\n        the losses are averaged over each loss element in the batch. Note that for\n        some losses, there are multiple elements per sample. If the field :attr:`size_average`\n        is set to ``False``, the losses are instead summed for each minibatch. Ignored\n        when reduce is ``False``.",
        default=True,
    )
    argument_parser.add_argument(
        "--reduce",
        type=bool,
        help="Deprecated (see :attr:`reduction`). By default, the\n        losses are averaged or summed over observations for each minibatch depending\n        on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per\n        batch element instead and ignores :attr:`size_average`.",
        default=True,
    )
    argument_parser.add_argument("--pos_weight", type=str, help=None)
    argument_parser.add_argument(
        "--reduction", type=str, help=None, required=True, default="mean"
    )
    return argument_parser


def CTCLossConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "The Connectionist Temporal Classification loss.\n\nCalculates loss between a continuous (unsegmented) time series and a target sequence. CTCLoss sums over the\nprobability of possible alignments of input to target, producing a loss value which is differentiable\nwith respect to each input node. The alignment of input to target is assumed to be \"many-to-one\", which\nlimits the length of the target sequence such that it must be :math:`\\leq` the input length.\n\n\n    reduction (string, optional): Specifies the reduction to apply to the output:\n            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,\n            ``'mean'``: the output losses will be divided by the target lengths and\n            then the mean over the batch is taken. Default: ``'mean'``\n    zero_infinity (bool, optional):\n            Whether to zero infinite losses and the associated gradients.\n            Default: ``False``\n            Infinite losses mainly occur when the inputs are too short\n            to be aligned to the targets.\n\n\nShape:\n    - Log_probs: Tensor of size :math:`(T, N, C)`,\n      where :math:`T = \\text{input length}`,\n      :math:`N = \\text{batch size}`, and\n      :math:`C = \\text{number of classes (including blank)}`.\n      The logarithmized probabilities of the outputs (e.g. obtained with\n      :func:`torch.nn.functional.log_softmax`).\n    - Targets: Tensor of size :math:`(N, S)` or\n      :math:`(\\operatorname{sum}(\\text{target\\_lengths}))`,\n      where :math:`N = \\text{batch size}` and\n      :math:`S = \\text{max target length, if shape is } (N, S)`.\n      It represent the target sequences. Each element in the target\n      sequence is a class index. And the target index cannot be blank (default=0).\n      In the :math:`(N, S)` form, targets are padded to the\n      length of the longest sequence, and stacked.\n      In the :math:`(\\operatorname{sum}(\\text{target\\_lengths}))` form,\n      the targets are assumed to be un-padded and\n      concatenated within 1 dimension.\n    - Input_lengths: Tuple or tensor of size :math:`(N)`,\n      where :math:`N = \\text{batch size}`. It represent the lengths of the\n      inputs (must each be :math:`\\leq T`). And the lengths are specified\n      for each sequence to achieve masking under the assumption that sequences\n      are padded to equal lengths.\n    - Target_lengths: Tuple or tensor of size :math:`(N)`,\n      where :math:`N = \\text{batch size}`. It represent lengths of the targets.\n      Lengths are specified for each sequence to achieve masking under the\n      assumption that sequences are padded to equal lengths. If target shape is\n      :math:`(N,S)`, target_lengths are effectively the stop index\n      :math:`s_n` for each target sequence, such that ``target_n = targets[n,0:s_n]`` for\n      each target in a batch. Lengths must each be :math:`\\leq S`\n      If the targets are given as a 1d tensor that is the concatenation of individual\n      targets, the target_lengths must add up to the total length of the tensor.\n    - Output: scalar. If :attr:`reduction` is ``'none'``, then\n      :math:`(N)`, where :math:`N = \\text{batch size}`.\n\nExamples::\n\n    >>> # Target are to be padded\n    >>> T = 50      # Input sequence length\n    >>> C = 20      # Number of classes (including blank)\n    >>> N = 16      # Batch size\n    >>> S = 30      # Target sequence length of longest target in batch (padding length)\n    >>> S_min = 10  # Minimum target length, for demonstration purposes\n    >>>\n    >>> # Initialize random batch of input vectors, for *size = (T,N,C)\n    >>> input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()\n    >>>\n    >>> # Initialize random batch of targets (0 = blank, 1:C = classes)\n    >>> target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.long)\n    >>>\n    >>> input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)\n    >>> target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)\n    >>> ctc_loss = nn.CTCLoss()\n    >>> loss = ctc_loss(input, target, input_lengths, target_lengths)\n    >>> loss.backward()\n    >>>\n    >>>\n    >>> # Target are to be un-padded\n    >>> T = 50      # Input sequence length\n    >>> C = 20      # Number of classes (including blank)\n    >>> N = 16      # Batch size\n    >>>\n    >>> # Initialize random batch of input vectors, for *size = (T,N,C)\n    >>> input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()\n    >>> input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)\n    >>>\n    >>> # Initialize random batch of targets (0 = blank, 1:C = classes)\n    >>> target_lengths = torch.randint(low=1, high=T, size=(N,), dtype=torch.long)\n    >>> target = torch.randint(low=1, high=C, size=(sum(target_lengths),), dtype=torch.long)\n    >>> ctc_loss = nn.CTCLoss()\n    >>> loss = ctc_loss(input, target, input_lengths, target_lengths)\n    >>> loss.backward()\n\nReference:\n    A. Graves et al.: Connectionist Temporal Classification:\n    Labelling Unsegmented Sequence Data with Recurrent Neural Networks:\n    https://www.cs.toronto.edu/~graves/icml_2006.pdf\n\nNote:\n    In order to use CuDNN, the following must be satisfied: :attr:`targets` must be\n    in concatenated format, all :attr:`input_lengths` must be `T`.  :math:`blank=0`,\n    :attr:`target_lengths` :math:`\\leq 256`, the integer arguments must be of\n    dtype :attr:`torch.int32`.\n\n    The regular implementation uses the (more common in PyTorch) `torch.long` dtype.\n\n\nNote:\n    In some circumstances when using the CUDA backend with CuDNN, this operator\n    may select a nondeterministic algorithm to increase performance. If this is\n    undesirable, you can try to make the operation deterministic (potentially at\n    a performance cost) by setting ``torch.backends.cudnn.deterministic =\n    True``.\n    Please see the notes on :doc:`/notes/randomness` for background."
    argument_parser.add_argument(
        "--blank",
        type=int,
        help="blank label. Default :math:`0`.",
        required=True,
        default=0,
    )
    argument_parser.add_argument(
        "--zero_infinity", type=bool, help="", required=True, default=False
    )
    argument_parser.add_argument(
        "--__constants__",
        type=loads,
        help="",
        required=True,
        default="['blank', 'reduction']",
    )
    argument_parser.add_argument(
        "--reduction", type=str, help=None, required=True, default="mean"
    )
    return argument_parser


def CosineEmbeddingLossConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Creates a criterion that measures the loss given input tensors\n:math:`x_1`, :math:`x_2` and a `Tensor` label :math:`y` with values 1 or -1.\nThis is used for measuring whether two inputs are similar or dissimilar,\nusing the cosine distance, and is typically used for learning nonlinear\nembeddings or semi-supervised learning.\n\nThe loss function for each sample is:\n\n.. math::\n    \\text{loss}(x, y) =\n    \\begin{cases}\n    1 - \\cos(x_1, x_2), & \\text{if } y = 1 \\\\\n    \\max(0, \\cos(x_1, x_2) - \\text{margin}), & \\text{if } y = -1\n    \\end{cases}\n\n\n    reduction (string, optional): Specifies the reduction to apply to the output:\n            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,\n            ``'mean'``: the sum of the output will be divided by the number of\n            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`\n            and :attr:`reduce` are in the process of being deprecated, and in the meantime,\n            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``"
    argument_parser.add_argument(
        "--margin",
        type=float,
        help="Should be a number from :math:`-1` to :math:`1`,\n        :math:`0` to :math:`0.5` is suggested. If :attr:`margin` is missing, the\n       ",
        required=True,
        default=0.0,
    )
    argument_parser.add_argument(
        "--size_average",
        type=bool,
        help="Deprecated (see :attr:`reduction`). By default,\n        the losses are averaged over each loss element in the batch. Note that for\n        some losses, there are multiple elements per sample. If the field :attr:`size_average`\n        is set to ``False``, the losses are instead summed for each minibatch. Ignored\n        when reduce is ``False``.",
        default=True,
    )
    argument_parser.add_argument(
        "--reduce",
        type=bool,
        help="Deprecated (see :attr:`reduction`). By default, the\n        losses are averaged or summed over observations for each minibatch depending\n        on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per\n        batch element instead and ignores :attr:`size_average`.",
        default=True,
    )
    argument_parser.add_argument(
        "--__constants__",
        type=loads,
        help="",
        required=True,
        default="['margin', 'reduction']",
    )
    argument_parser.add_argument(
        "--reduction", type=str, help=None, required=True, default="mean"
    )
    return argument_parser


def CrossEntropyLossConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "This criterion combines :func:`nn.LogSoftmax` and :func:`nn.NLLLoss` in one single class.\n\nIt is useful when training a classification problem with `C` classes.\nIf provided, the optional argument :attr:`weight` should be a 1D `Tensor`\nassigning weight to each of the classes.\nThis is particularly useful when you have an unbalanced training set.\n\nThe `input` is expected to contain raw, unnormalized scores for each class.\n\n`input` has to be a Tensor of size either :math:`(minibatch, C)` or\n:math:`(minibatch, C, d_1, d_2, ..., d_K)`\nwith :math:`K \\geq 1` for the `K`-dimensional case (described later).\n\nThis criterion expects a class index in the range :math:`[0, C-1]` as the\n`target` for each value of a 1D tensor of size `minibatch`; if `ignore_index`\nis specified, this criterion also accepts this class index (this index may not\nnecessarily be in the class range).\n\nThe loss can be described as:\n\n.. math::\n    \\text{loss}(x, class) = -\\log\\left(\\frac{\\exp(x[class])}{\\sum_j \\exp(x[j])}\\right)\n                   = -x[class] + \\log\\left(\\sum_j \\exp(x[j])\\right)\n\nor in the case of the :attr:`weight` argument being specified:\n\n.. math::\n    \\text{loss}(x, class) = weight[class] \\left(-x[class] + \\log\\left(\\sum_j \\exp(x[j])\\right)\\right)\n\nThe losses are averaged across observations for each minibatch. If the\n:attr:`weight` argument is specified then this is a weighted average:\n\n.. math::\n    \\text{loss} = \\frac{\\sum^{N}_{i=1} loss(i, class[i])}{\\sum^{N}_{i=1} weight[class[i]]}\n\nCan also be used for higher dimension inputs, such as 2D images, by providing\nan input of size :math:`(minibatch, C, d_1, d_2, ..., d_K)` with :math:`K \\geq 1`,\nwhere :math:`K` is the number of dimensions, and a target of appropriate shape\n(see below).\n\n\n    reduction (string, optional): Specifies the reduction to apply to the output:\n            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will\n            be applied, ``'mean'``: the weighted mean of the output is taken,\n            ``'sum'``: the output will be summed. Note: :attr:`size_average`\n            and :attr:`reduce` are in the process of being deprecated, and in\n            the meantime, specifying either of those two args will override\n            :attr:`reduction`. Default: ``'mean'``\n\n\nShape:\n    - Input: :math:`(N, C)` where `C = number of classes`, or\n      :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \\geq 1`\n      in the case of `K`-dimensional loss.\n    - Target: :math:`(N)` where each value is :math:`0 \\leq \\text{targets}[i] \\leq C-1`, or\n      :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \\geq 1` in the case of\n      K-dimensional loss.\n    - Output: scalar.\n      If :attr:`reduction` is ``'none'``, then the same size as the target:\n      :math:`(N)`, or\n      :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \\geq 1` in the case\n      of K-dimensional loss.\n\nExamples::\n\n    >>> loss = nn.CrossEntropyLoss()\n    >>> input = torch.randn(3, 5, requires_grad=True)\n    >>> target = torch.empty(3, dtype=torch.long).random_(5)\n    >>> output = loss(input, target)\n    >>> output.backward()"
    argument_parser.add_argument(
        "--weight",
        type=str,
        help="a manual rescaling weight given to each class.\n        If given, has to be a Tensor of size `C`",
    )
    argument_parser.add_argument(
        "--size_average",
        type=bool,
        help="Deprecated (see :attr:`reduction`). By default,\n        the losses are averaged over each loss element in the batch. Note that for\n        some losses, there are multiple elements per sample. If the field :attr:`size_average`\n        is set to ``False``, the losses are instead summed for each minibatch. Ignored\n        when reduce is ``False``.",
        default=True,
    )
    argument_parser.add_argument(
        "--ignore_index",
        type=int,
        help="Specifies a target value that is ignored\n        and does not contribute to the input gradient. When :attr:`size_average` is\n        ``True``, the loss is averaged over non-ignored targets.",
        required=True,
        default=-100,
    )
    argument_parser.add_argument(
        "--reduce",
        type=bool,
        help="Deprecated (see :attr:`reduction`). By default, the\n        losses are averaged or summed over observations for each minibatch depending\n        on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per\n        batch element instead and ignores :attr:`size_average`.",
        default=True,
    )
    argument_parser.add_argument(
        "--__constants__",
        type=loads,
        help="",
        required=True,
        default="['ignore_index', 'reduction']",
    )
    argument_parser.add_argument(
        "--reduction", type=str, help=None, required=True, default="mean"
    )
    return argument_parser


def HingeEmbeddingLossConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Measures the loss given an input tensor :math:`x` and a labels tensor :math:`y`\n(containing 1 or -1).\nThis is usually used for measuring whether two inputs are similar or\ndissimilar, e.g. using the L1 pairwise distance as :math:`x`, and is typically\nused for learning nonlinear embeddings or semi-supervised learning.\n\nThe loss function for :math:`n`-th sample in the mini-batch is\n\n.. math::\n    l_n = \\begin{cases}\n        x_n, & \\text{if}\\; y_n = 1,\\\\\n        \\max \\{0, \\Delta - x_n\\}, & \\text{if}\\; y_n = -1,\n    \\end{cases}\n\nand the total loss functions is\n\n.. math::\n    \\ell(x, y) = \\begin{cases}\n        \\operatorname{mean}(L), & \\text{if reduction} = \\text{'mean';}\\\\\n        \\operatorname{sum}(L),  & \\text{if reduction} = \\text{'sum'.}\n    \\end{cases}\n\nwhere :math:`L = \\{l_1,\\dots,l_N\\}^\\top`.\n\n\n    reduction (string, optional): Specifies the reduction to apply to the output:\n            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,\n            ``'mean'``: the sum of the output will be divided by the number of\n            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`\n            and :attr:`reduce` are in the process of being deprecated, and in the meantime,\n            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``\n\n\nShape:\n    - Input: :math:`(*)` where :math:`*` means, any number of dimensions. The sum operation\n      operates over all the elements.\n    - Target: :math:`(*)`, same shape as the input\n    - Output: scalar. If :attr:`reduction` is ``'none'``, then same shape as the input"
    argument_parser.add_argument(
        "--margin",
        type=float,
        help="Has a default value of `1`.",
        required=True,
        default=1.0,
    )
    argument_parser.add_argument(
        "--size_average",
        type=bool,
        help="Deprecated (see :attr:`reduction`). By default,\n        the losses are averaged over each loss element in the batch. Note that for\n        some losses, there are multiple elements per sample. If the field :attr:`size_average`\n        is set to ``False``, the losses are instead summed for each minibatch. Ignored\n        when reduce is ``False``.",
        default=True,
    )
    argument_parser.add_argument(
        "--reduce",
        type=bool,
        help="Deprecated (see :attr:`reduction`). By default, the\n        losses are averaged or summed over observations for each minibatch depending\n        on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per\n        batch element instead and ignores :attr:`size_average`.",
        default=True,
    )
    argument_parser.add_argument(
        "--__constants__",
        type=loads,
        help="",
        required=True,
        default="['margin', 'reduction']",
    )
    argument_parser.add_argument(
        "--reduction", type=str, help=None, required=True, default="mean"
    )
    return argument_parser


def KLDivLossConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "The Kullback-Leibler divergence loss measure\n\n`Kullback-Leibler divergence`_ is a useful distance measure for continuous\ndistributions and is often useful when performing direct regression over\nthe space of (discretely sampled) continuous output distributions.\n\nAs with :class:`~torch.nn.NLLLoss`, the `input` given is expected to contain\n*log-probabilities* and is not restricted to a 2D Tensor.\nThe targets are interpreted as *probabilities* by default, but could be considered\nas *log-probabilities* with :attr:`log_target` set to ``True``.\n\nThis criterion expects a `target` `Tensor` of the same size as the\n`input` `Tensor`.\n\nThe unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:\n\n.. math::\n    l(x,y) = L = \\{ l_1,\\dots,l_N \\}, \\quad\n    l_n = y_n \\cdot \\left( \\log y_n - x_n \\right)\n\nwhere the index :math:`N` spans all dimensions of ``input`` and :math:`L` has the same\nshape as ``input``. If :attr:`reduction` is not ``'none'`` (default ``'mean'``), then:\n\n.. math::\n    \\ell(x, y) = \\begin{cases}\n        \\operatorname{mean}(L), & \\text{if reduction} = \\text{'mean';} \\\\\n        \\operatorname{sum}(L),  & \\text{if reduction} = \\text{'sum'.}\n    \\end{cases}\n\nIn default :attr:`reduction` mode ``'mean'``, the losses are averaged for each minibatch over observations\n**as well as** over dimensions. ``'batchmean'`` mode gives the correct KL divergence where losses\nare averaged over batch dimension only. ``'mean'`` mode's behavior will be changed to the same as\n``'batchmean'`` in the next major release.\n\n.. _`kullback-leibler divergence`: https://en.wikipedia.org/wiki/Kullback-Leibler_divergence\n\n\n    reduction (string, optional): Specifies the reduction to apply to the output:\n            ``'none'`` | ``'batchmean'`` | ``'sum'`` | ``'mean'``.\n            ``'none'``: no reduction will be applied.\n            ``'batchmean'``: the sum of the output will be divided by batchsize.\n            ``'sum'``: the output will be summed.\n            ``'mean'``: the output will be divided by the number of elements in the output.\n            Default: ``'mean'``\n        log_target (bool, optional): Specifies whether `target` is passed in the log space.\n            Default: ``False``\n\n\n.. note::\n    :attr:`size_average` and :attr:`reduce` are in the process of being deprecated,\n    and in the meantime, specifying either of those two args will override :attr:`reduction`.\n\n.. note::\n    :attr:`reduction` = ``'mean'`` doesn't return the true kl divergence value, please use\n    :attr:`reduction` = ``'batchmean'`` which aligns with KL math definition.\n    In the next major release, ``'mean'`` will be changed to be the same as ``'batchmean'``.\n\nShape:\n    - Input: :math:`(N, *)` where :math:`*` means, any number of additional\n      dimensions\n    - Target: :math:`(N, *)`, same shape as the input\n    - Output: scalar by default. If :attr:``reduction`` is ``'none'``, then :math:`(N, *)`,\n      the same shape as the input"
    argument_parser.add_argument(
        "--size_average",
        type=bool,
        help="Deprecated (see :attr:`reduction`). By default,\n        the losses are averaged over each loss element in the batch. Note that for\n        some losses, there are multiple elements per sample. If the field :attr:`size_average`\n        is set to ``False``, the losses are instead summed for each minibatch. Ignored\n        when reduce is ``False``.",
        default=True,
    )
    argument_parser.add_argument(
        "--reduce",
        type=bool,
        help="Deprecated (see :attr:`reduction`). By default, the\n        losses are averaged or summed over observations for each minibatch depending\n        on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per\n        batch element instead and ignores :attr:`size_average`.",
        default=True,
    )
    argument_parser.add_argument(
        "--__constants__",
        type=str,
        action="append",
        help="",
        required=True,
        default="reduction",
    )
    argument_parser.add_argument(
        "--log_target", type=bool, help=None, required=True, default=False
    )
    argument_parser.add_argument(
        "--reduction", type=str, help=None, required=True, default="mean"
    )
    return argument_parser


def L1LossConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Creates a criterion that measures the mean absolute error (MAE) between each element in\nthe input :math:`x` and target :math:`y`.\n\nThe unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:\n\n.. math::\n    \\ell(x, y) = L = \\{l_1,\\dots,l_N\\}^\\top, \\quad\n    l_n = \\left| x_n - y_n \\right|,\n\nwhere :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``\n(default ``'mean'``), then:\n\n.. math::\n    \\ell(x, y) =\n    \\begin{cases}\n        \\operatorname{mean}(L), & \\text{if reduction} = \\text{'mean';}\\\\\n        \\operatorname{sum}(L),  & \\text{if reduction} = \\text{'sum'.}\n    \\end{cases}\n\n:math:`x` and :math:`y` are tensors of arbitrary shapes with a total\nof :math:`n` elements each.\n\nThe sum operation still operates over all the elements, and divides by :math:`n`.\n\nThe division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.\n\n\n    reduction (string, optional): Specifies the reduction to apply to the output:\n            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,\n            ``'mean'``: the sum of the output will be divided by the number of\n            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`\n            and :attr:`reduce` are in the process of being deprecated, and in the meantime,\n            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``\n\n\nShape:\n    - Input: :math:`(N, *)` where :math:`*` means, any number of additional\n      dimensions\n    - Target: :math:`(N, *)`, same shape as the input\n    - Output: scalar. If :attr:`reduction` is ``'none'``, then\n      :math:`(N, *)`, same shape as the input\n\nExamples::\n\n    >>> loss = nn.L1Loss()\n    >>> input = torch.randn(3, 5, requires_grad=True)\n    >>> target = torch.randn(3, 5)\n    >>> output = loss(input, target)\n    >>> output.backward()"
    argument_parser.add_argument(
        "--size_average",
        type=bool,
        help="Deprecated (see :attr:`reduction`). By default,\n        the losses are averaged over each loss element in the batch. Note that for\n        some losses, there are multiple elements per sample. If the field :attr:`size_average`\n        is set to ``False``, the losses are instead summed for each minibatch. Ignored\n        when reduce is ``False``.",
        default=True,
    )
    argument_parser.add_argument(
        "--reduce",
        type=bool,
        help="Deprecated (see :attr:`reduction`). By default, the\n        losses are averaged or summed over observations for each minibatch depending\n        on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per\n        batch element instead and ignores :attr:`size_average`.",
        default=True,
    )
    argument_parser.add_argument(
        "--__constants__",
        type=str,
        action="append",
        help="",
        required=True,
        default="reduction",
    )
    argument_parser.add_argument(
        "--reduction", type=str, help=None, required=True, default="mean"
    )
    return argument_parser


def MSELossConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Creates a criterion that measures the mean squared error (squared L2 norm) between\neach element in the input :math:`x` and target :math:`y`.\n\nThe unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:\n\n.. math::\n    \\ell(x, y) = L = \\{l_1,\\dots,l_N\\}^\\top, \\quad\n    l_n = \\left( x_n - y_n \\right)^2,\n\nwhere :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``\n(default ``'mean'``), then:\n\n.. math::\n    \\ell(x, y) =\n    \\begin{cases}\n        \\operatorname{mean}(L), &  \\text{if reduction} = \\text{'mean';}\\\\\n        \\operatorname{sum}(L),  &  \\text{if reduction} = \\text{'sum'.}\n    \\end{cases}\n\n:math:`x` and :math:`y` are tensors of arbitrary shapes with a total\nof :math:`n` elements each.\n\nThe mean operation still operates over all the elements, and divides by :math:`n`.\n\nThe division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.\n\n\n    reduction (string, optional): Specifies the reduction to apply to the output:\n            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,\n            ``'mean'``: the sum of the output will be divided by the number of\n            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`\n            and :attr:`reduce` are in the process of being deprecated, and in the meantime,\n            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``\n\n\nShape:\n    - Input: :math:`(N, *)` where :math:`*` means, any number of additional\n      dimensions\n    - Target: :math:`(N, *)`, same shape as the input\n\nExamples::\n\n    >>> loss = nn.MSELoss()\n    >>> input = torch.randn(3, 5, requires_grad=True)\n    >>> target = torch.randn(3, 5)\n    >>> output = loss(input, target)\n    >>> output.backward()"
    argument_parser.add_argument(
        "--size_average",
        type=bool,
        help="Deprecated (see :attr:`reduction`). By default,\n        the losses are averaged over each loss element in the batch. Note that for\n        some losses, there are multiple elements per sample. If the field :attr:`size_average`\n        is set to ``False``, the losses are instead summed for each minibatch. Ignored\n        when reduce is ``False``.",
        default=True,
    )
    argument_parser.add_argument(
        "--reduce",
        type=bool,
        help="Deprecated (see :attr:`reduction`). By default, the\n        losses are averaged or summed over observations for each minibatch depending\n        on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per\n        batch element instead and ignores :attr:`size_average`.",
        default=True,
    )
    argument_parser.add_argument(
        "--__constants__",
        type=str,
        action="append",
        help="",
        required=True,
        default="reduction",
    )
    argument_parser.add_argument(
        "--reduction", type=str, help=None, required=True, default="mean"
    )
    return argument_parser


def MarginRankingLossConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Creates a criterion that measures the loss given\ninputs :math:`x1`, :math:`x2`, two 1D mini-batch `Tensors`,\nand a label 1D mini-batch tensor :math:`y` (containing 1 or -1).\n\nIf :math:`y = 1` then it assumed the first input should be ranked higher\n(have a larger value) than the second input, and vice-versa for :math:`y = -1`.\n\nThe loss function for each pair of samples in the mini-batch is:\n\n.. math::\n    \\text{loss}(x1, x2, y) = \\max(0, -y * (x1 - x2) + \\text{margin})\n\n\n    reduction (string, optional): Specifies the reduction to apply to the output:\n            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,\n            ``'mean'``: the sum of the output will be divided by the number of\n            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`\n            and :attr:`reduce` are in the process of being deprecated, and in the meantime,\n            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``\n\n\nShape:\n    - Input1: :math:`(N)` where `N` is the batch size.\n    - Input2: :math:`(N)`, same shape as the Input1.\n    - Target: :math:`(N)`, same shape as the inputs.\n    - Output: scalar. If :attr:`reduction` is ``'none'``, then :math:`(N)`.\n\nExamples::\n\n    >>> loss = nn.MarginRankingLoss()\n    >>> input1 = torch.randn(3, requires_grad=True)\n    >>> input2 = torch.randn(3, requires_grad=True)\n    >>> target = torch.randn(3).sign()\n    >>> output = loss(input1, input2, target)\n    >>> output.backward()"
    argument_parser.add_argument(
        "--margin",
        type=float,
        help="Has a default value of :math:`0`.",
        required=True,
        default=0.0,
    )
    argument_parser.add_argument(
        "--size_average",
        type=bool,
        help="Deprecated (see :attr:`reduction`). By default,\n        the losses are averaged over each loss element in the batch. Note that for\n        some losses, there are multiple elements per sample. If the field :attr:`size_average`\n        is set to ``False``, the losses are instead summed for each minibatch. Ignored\n        when reduce is ``False``.",
        default=True,
    )
    argument_parser.add_argument(
        "--reduce",
        type=bool,
        help="Deprecated (see :attr:`reduction`). By default, the\n        losses are averaged or summed over observations for each minibatch depending\n        on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per\n        batch element instead and ignores :attr:`size_average`.",
        default=True,
    )
    argument_parser.add_argument(
        "--__constants__",
        type=loads,
        help="",
        required=True,
        default="['margin', 'reduction']",
    )
    argument_parser.add_argument(
        "--reduction", type=str, help=None, required=True, default="mean"
    )
    return argument_parser


def MultiLabelMarginLossConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Creates a criterion that optimizes a multi-class multi-classification\nhinge loss (margin-based loss) between input :math:`x` (a 2D mini-batch `Tensor`)\nand output :math:`y` (which is a 2D `Tensor` of target class indices).\nFor each sample in the mini-batch:\n\n.. math::\n    \\text{loss}(x, y) = \\sum_{ij}\\frac{\\max(0, 1 - (x[y[j]] - x[i]))}{\\text{x.size}(0)}\n\nwhere :math:`x \\in \\left\\{0, \\; \\cdots , \\; \\text{x.size}(0) - 1\\right\\}`, \\\n:math:`y \\in \\left\\{0, \\; \\cdots , \\; \\text{y.size}(0) - 1\\right\\}`, \\\n:math:`0 \\leq y[j] \\leq \\text{x.size}(0)-1`, \\\nand :math:`i \\neq y[j]` for all :math:`i` and :math:`j`.\n\n:math:`y` and :math:`x` must have the same size.\n\nThe criterion only considers a contiguous block of non-negative targets that\nstarts at the front.\n\nThis allows for different samples to have variable amounts of target classes.\n\n\n    reduction (string, optional): Specifies the reduction to apply to the output:\n            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,\n            ``'mean'``: the sum of the output will be divided by the number of\n            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`\n            and :attr:`reduce` are in the process of being deprecated, and in the meantime,\n            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``\n\n\nShape:\n    - Input: :math:`(C)` or :math:`(N, C)` where `N` is the batch size and `C`\n      is the number of classes.\n    - Target: :math:`(C)` or :math:`(N, C)`, label targets padded by -1 ensuring same shape as the input.\n    - Output: scalar. If :attr:`reduction` is ``'none'``, then :math:`(N)`.\n\nExamples::\n\n    >>> loss = nn.MultiLabelMarginLoss()\n    >>> x = torch.FloatTensor([[0.1, 0.2, 0.4, 0.8]])\n    >>> # for target y, only consider labels 3 and 0, not after label -1\n    >>> y = torch.LongTensor([[3, 0, -1, 1]])\n    >>> loss(x, y)\n    >>> # 0.25 * ((1-(0.1-0.2)) + (1-(0.1-0.4)) + (1-(0.8-0.2)) + (1-(0.8-0.4)))\n    tensor(0.8500)"
    argument_parser.add_argument(
        "--size_average",
        type=bool,
        help="Deprecated (see :attr:`reduction`). By default,\n        the losses are averaged over each loss element in the batch. Note that for\n        some losses, there are multiple elements per sample. If the field :attr:`size_average`\n        is set to ``False``, the losses are instead summed for each minibatch. Ignored\n        when reduce is ``False``.",
        default=True,
    )
    argument_parser.add_argument(
        "--reduce",
        type=bool,
        help="Deprecated (see :attr:`reduction`). By default, the\n        losses are averaged or summed over observations for each minibatch depending\n        on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per\n        batch element instead and ignores :attr:`size_average`.",
        default=True,
    )
    argument_parser.add_argument(
        "--__constants__",
        type=str,
        action="append",
        help="",
        required=True,
        default="reduction",
    )
    argument_parser.add_argument(
        "--reduction", type=str, help=None, required=True, default="mean"
    )
    return argument_parser


def MultiLabelSoftMarginLossConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Creates a criterion that optimizes a multi-label one-versus-all\nloss based on max-entropy, between input :math:`x` and target :math:`y` of size\n:math:`(N, C)`.\nFor each sample in the minibatch:\n\n.. math::\n    loss(x, y) = - \\frac{1}{C} * \\sum_i y[i] * \\log((1 + \\exp(-x[i]))^{-1})\n                     + (1-y[i]) * \\log\\left(\\frac{\\exp(-x[i])}{(1 + \\exp(-x[i]))}\\right)\n\nwhere :math:`i \\in \\left\\{0, \\; \\cdots , \\; \\text{x.nElement}() - 1\\right\\}`,\n:math:`y[i] \\in \\left\\{0, \\; 1\\right\\}`.\n\n\n    reduction (string, optional): Specifies the reduction to apply to the output:\n            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,\n            ``'mean'``: the sum of the output will be divided by the number of\n            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`\n            and :attr:`reduce` are in the process of being deprecated, and in the meantime,\n            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``\n\n\nShape:\n    - Input: :math:`(N, C)` where `N` is the batch size and `C` is the number of classes.\n    - Target: :math:`(N, C)`, label targets padded by -1 ensuring same shape as the input.\n    - Output: scalar. If :attr:`reduction` is ``'none'``, then :math:`(N)`."
    argument_parser.add_argument(
        "--weight",
        type=str,
        help="a manual rescaling weight given to each\n        class. If given, it has to be a Tensor of size `C`. Otherwise, it is\n        treated as if having all ones.",
    )
    argument_parser.add_argument(
        "--size_average",
        type=bool,
        help="Deprecated (see :attr:`reduction`). By default,\n        the losses are averaged over each loss element in the batch. Note that for\n        some losses, there are multiple elements per sample. If the field :attr:`size_average`\n        is set to ``False``, the losses are instead summed for each minibatch. Ignored\n        when reduce is ``False``.",
        default=True,
    )
    argument_parser.add_argument(
        "--reduce",
        type=bool,
        help="Deprecated (see :attr:`reduction`). By default, the\n        losses are averaged or summed over observations for each minibatch depending\n        on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per\n        batch element instead and ignores :attr:`size_average`.",
        default=True,
    )
    argument_parser.add_argument(
        "--__constants__",
        type=str,
        action="append",
        help="",
        required=True,
        default="reduction",
    )
    argument_parser.add_argument(
        "--reduction", type=str, help=None, required=True, default="mean"
    )
    return argument_parser


def MultiMarginLossConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Creates a criterion that optimizes a multi-class classification hinge\nloss (margin-based loss) between input :math:`x` (a 2D mini-batch `Tensor`) and\noutput :math:`y` (which is a 1D tensor of target class indices,\n:math:`0 \\leq y \\leq \\text{x.size}(1)-1`):\n\nFor each mini-batch sample, the loss in terms of the 1D input :math:`x` and scalar\noutput :math:`y` is:\n\n.. math::\n    \\text{loss}(x, y) = \\frac{\\sum_i \\max(0, \\text{margin} - x[y] + x[i]))^p}{\\text{x.size}(0)}\n\nwhere :math:`x \\in \\left\\{0, \\; \\cdots , \\; \\text{x.size}(0) - 1\\right\\}`\nand :math:`i \\neq y`.\n\nOptionally, you can give non-equal weighting on the classes by passing\na 1D :attr:`weight` tensor into the constructor.\n\nThe loss function then becomes:\n\n.. math::\n    \\text{loss}(x, y) = \\frac{\\sum_i \\max(0, w[y] * (\\text{margin} - x[y] + x[i]))^p)}{\\text{x.size}(0)}\n\n\n    reduction (string, optional): Specifies the reduction to apply to the output:\n            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,\n            ``'mean'``: the sum of the output will be divided by the number of\n            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`\n            and :attr:`reduce` are in the process of being deprecated, and in the meantime,\n            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``"
    argument_parser.add_argument(
        "--p",
        type=int,
        help="Has a default value of :math:`1`. :math:`1` and :math:`2`\n        are the only supported values.",
        required=True,
        default=1,
    )
    argument_parser.add_argument(
        "--margin",
        type=float,
        help="Has a default value of :math:`1`.",
        required=True,
        default=1.0,
    )
    argument_parser.add_argument(
        "--weight",
        type=str,
        help="a manual rescaling weight given to each\n        class. If given, it has to be a Tensor of size `C`. Otherwise, it is\n        treated as if having all ones.",
    )
    argument_parser.add_argument(
        "--size_average",
        type=bool,
        help="Deprecated (see :attr:`reduction`). By default,\n        the losses are averaged over each loss element in the batch. Note that for\n        some losses, there are multiple elements per sample. If the field :attr:`size_average`\n        is set to ``False``, the losses are instead summed for each minibatch. Ignored\n        when reduce is ``False``.",
        default=True,
    )
    argument_parser.add_argument(
        "--reduce",
        type=bool,
        help="Deprecated (see :attr:`reduction`). By default, the\n        losses are averaged or summed over observations for each minibatch depending\n        on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per\n        batch element instead and ignores :attr:`size_average`.",
        default=True,
    )
    argument_parser.add_argument(
        "--__constants__",
        type=loads,
        help="",
        required=True,
        default="['p', 'margin', 'reduction']",
    )
    argument_parser.add_argument(
        "--reduction", type=str, help=None, required=True, default="mean"
    )
    return argument_parser


def NLLLossConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "The negative log likelihood loss. It is useful to train a classification\nproblem with `C` classes.\n\nIf provided, the optional argument :attr:`weight` should be a 1D Tensor assigning\nweight to each of the classes. This is particularly useful when you have an\nunbalanced training set.\n\nThe `input` given through a forward call is expected to contain\nlog-probabilities of each class. `input` has to be a Tensor of size either\n:math:`(minibatch, C)` or :math:`(minibatch, C, d_1, d_2, ..., d_K)`\nwith :math:`K \\geq 1` for the `K`-dimensional case (described later).\n\nObtaining log-probabilities in a neural network is easily achieved by\nadding a  `LogSoftmax`  layer in the last layer of your network.\nYou may use `CrossEntropyLoss` instead, if you prefer not to add an extra\nlayer.\n\nThe `target` that this loss expects should be a class index in the range :math:`[0, C-1]`\nwhere `C = number of classes`; if `ignore_index` is specified, this loss also accepts\nthis class index (this index may not necessarily be in the class range).\n\nThe unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:\n\n.. math::\n    \\ell(x, y) = L = \\{l_1,\\dots,l_N\\}^\\top, \\quad\n    l_n = - w_{y_n} x_{n,y_n}, \\quad\n    w_{c} = \\text{weight}[c] \\cdot \\mathbb{1}\\{c \\not= \\text{ignore\\_index}\\},\n\nwhere :math:`x` is the input, :math:`y` is the target, :math:`w` is the weight, and\n:math:`N` is the batch size. If :attr:`reduction` is not ``'none'``\n(default ``'mean'``), then\n\n.. math::\n    \\ell(x, y) = \\begin{cases}\n        \\sum_{n=1}^N \\frac{1}{\\sum_{n=1}^N w_{y_n}} l_n, &\n        \\text{if reduction} = \\text{'mean';}\\\\\n        \\sum_{n=1}^N l_n,  &\n        \\text{if reduction} = \\text{'sum'.}\n    \\end{cases}\n\nCan also be used for higher dimension inputs, such as 2D images, by providing\nan input of size :math:`(minibatch, C, d_1, d_2, ..., d_K)` with :math:`K \\geq 1`,\nwhere :math:`K` is the number of dimensions, and a target of appropriate shape\n(see below). In the case of images, it computes NLL loss per-pixel.\n\n\n    reduction (string, optional): Specifies the reduction to apply to the output:\n            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will\n            be applied, ``'mean'``: the weighted mean of the output is taken,\n            ``'sum'``: the output will be summed. Note: :attr:`size_average`\n            and :attr:`reduce` are in the process of being deprecated, and in\n            the meantime, specifying either of those two args will override\n            :attr:`reduction`. Default: ``'mean'``\n\n\nShape:\n    - Input: :math:`(N, C)` where `C = number of classes`, or\n      :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \\geq 1`\n      in the case of `K`-dimensional loss.\n    - Target: :math:`(N)` where each value is :math:`0 \\leq \\text{targets}[i] \\leq C-1`, or\n      :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \\geq 1` in the case of\n      K-dimensional loss.\n    - Output: scalar.\n      If :attr:`reduction` is ``'none'``, then the same size as the target: :math:`(N)`, or\n      :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \\geq 1` in the case\n      of K-dimensional loss.\n\nExamples::\n\n    >>> m = nn.LogSoftmax(dim=1)\n    >>> loss = nn.NLLLoss()\n    >>> # input is of size N x C = 3 x 5\n    >>> input = torch.randn(3, 5, requires_grad=True)\n    >>> # each element in target has to have 0 <= value < C\n    >>> target = torch.tensor([1, 0, 4])\n    >>> output = loss(m(input), target)\n    >>> output.backward()\n    >>>\n    >>>\n    >>> # 2D loss example (used, for example, with image inputs)\n    >>> N, C = 5, 4\n    >>> loss = nn.NLLLoss()\n    >>> # input is of size N x C x height x width\n    >>> data = torch.randn(N, 16, 10, 10)\n    >>> conv = nn.Conv2d(16, C, (3, 3))\n    >>> m = nn.LogSoftmax(dim=1)\n    >>> # each element in target has to have 0 <= value < C\n    >>> target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)\n    >>> output = loss(m(conv(data)), target)\n    >>> output.backward()"
    argument_parser.add_argument(
        "--weight",
        type=str,
        help="a manual rescaling weight given to each\n        class. If given, it has to be a Tensor of size `C`. Otherwise, it is\n        treated as if having all ones.",
    )
    argument_parser.add_argument(
        "--size_average",
        type=bool,
        help="Deprecated (see :attr:`reduction`). By default,\n        the losses are averaged over each loss element in the batch. Note that for\n        some losses, there are multiple elements per sample. If the field :attr:`size_average`\n        is set to ``False``, the losses are instead summed for each minibatch. Ignored\n        when reduce is ``False``.",
        default=True,
    )
    argument_parser.add_argument(
        "--ignore_index",
        type=int,
        help="Specifies a target value that is ignored\n        and does not contribute to the input gradient. When\n        :attr:`size_average` is ``True``, the loss is averaged over\n        non-ignored targets.",
        required=True,
        default=-100,
    )
    argument_parser.add_argument(
        "--reduce",
        type=bool,
        help="Deprecated (see :attr:`reduction`). By default, the\n        losses are averaged or summed over observations for each minibatch depending\n        on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per\n        batch element instead and ignores :attr:`size_average`.",
        default=True,
    )
    argument_parser.add_argument(
        "--__constants__",
        type=loads,
        help="",
        required=True,
        default="['ignore_index', 'reduction']",
    )
    argument_parser.add_argument(
        "--reduction", type=str, help=None, required=True, default="mean"
    )
    return argument_parser


def NLLLoss2dConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "The negative log likelihood loss. It is useful to train a classification\nproblem with `C` classes.\n\nIf provided, the optional argument :attr:`weight` should be a 1D Tensor assigning\nweight to each of the classes. This is particularly useful when you have an\nunbalanced training set.\n\nThe `input` given through a forward call is expected to contain\nlog-probabilities of each class. `input` has to be a Tensor of size either\n:math:`(minibatch, C)` or :math:`(minibatch, C, d_1, d_2, ..., d_K)`\nwith :math:`K \\geq 1` for the `K`-dimensional case (described later).\n\nObtaining log-probabilities in a neural network is easily achieved by\nadding a  `LogSoftmax`  layer in the last layer of your network.\nYou may use `CrossEntropyLoss` instead, if you prefer not to add an extra\nlayer.\n\nThe `target` that this loss expects should be a class index in the range :math:`[0, C-1]`\nwhere `C = number of classes`; if `ignore_index` is specified, this loss also accepts\nthis class index (this index may not necessarily be in the class range).\n\nThe unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:\n\n.. math::\n    \\ell(x, y) = L = \\{l_1,\\dots,l_N\\}^\\top, \\quad\n    l_n = - w_{y_n} x_{n,y_n}, \\quad\n    w_{c} = \\text{weight}[c] \\cdot \\mathbb{1}\\{c \\not= \\text{ignore\\_index}\\},\n\nwhere :math:`x` is the input, :math:`y` is the target, :math:`w` is the weight, and\n:math:`N` is the batch size. If :attr:`reduction` is not ``'none'``\n(default ``'mean'``), then\n\n.. math::\n    \\ell(x, y) = \\begin{cases}\n        \\sum_{n=1}^N \\frac{1}{\\sum_{n=1}^N w_{y_n}} l_n, &\n        \\text{if reduction} = \\text{'mean';}\\\\\n        \\sum_{n=1}^N l_n,  &\n        \\text{if reduction} = \\text{'sum'.}\n    \\end{cases}\n\nCan also be used for higher dimension inputs, such as 2D images, by providing\nan input of size :math:`(minibatch, C, d_1, d_2, ..., d_K)` with :math:`K \\geq 1`,\nwhere :math:`K` is the number of dimensions, and a target of appropriate shape\n(see below). In the case of images, it computes NLL loss per-pixel.\n\n\n    reduction (string, optional): Specifies the reduction to apply to the output:\n            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will\n            be applied, ``'mean'``: the weighted mean of the output is taken,\n            ``'sum'``: the output will be summed. Note: :attr:`size_average`\n            and :attr:`reduce` are in the process of being deprecated, and in\n            the meantime, specifying either of those two args will override\n            :attr:`reduction`. Default: ``'mean'``\n\n\nShape:\n    - Input: :math:`(N, C)` where `C = number of classes`, or\n      :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \\geq 1`\n      in the case of `K`-dimensional loss.\n    - Target: :math:`(N)` where each value is :math:`0 \\leq \\text{targets}[i] \\leq C-1`, or\n      :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \\geq 1` in the case of\n      K-dimensional loss.\n    - Output: scalar.\n      If :attr:`reduction` is ``'none'``, then the same size as the target: :math:`(N)`, or\n      :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \\geq 1` in the case\n      of K-dimensional loss.\n\nExamples::\n\n    >>> m = nn.LogSoftmax(dim=1)\n    >>> loss = nn.NLLLoss()\n    >>> # input is of size N x C = 3 x 5\n    >>> input = torch.randn(3, 5, requires_grad=True)\n    >>> # each element in target has to have 0 <= value < C\n    >>> target = torch.tensor([1, 0, 4])\n    >>> output = loss(m(input), target)\n    >>> output.backward()\n    >>>\n    >>>\n    >>> # 2D loss example (used, for example, with image inputs)\n    >>> N, C = 5, 4\n    >>> loss = nn.NLLLoss()\n    >>> # input is of size N x C x height x width\n    >>> data = torch.randn(N, 16, 10, 10)\n    >>> conv = nn.Conv2d(16, C, (3, 3))\n    >>> m = nn.LogSoftmax(dim=1)\n    >>> # each element in target has to have 0 <= value < C\n    >>> target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)\n    >>> output = loss(m(conv(data)), target)\n    >>> output.backward()"
    argument_parser.add_argument(
        "--weight",
        type=str,
        help="a manual rescaling weight given to each\n        class. If given, it has to be a Tensor of size `C`. Otherwise, it is\n        treated as if having all ones.",
    )
    argument_parser.add_argument(
        "--size_average",
        type=bool,
        help="Deprecated (see :attr:`reduction`). By default,\n        the losses are averaged over each loss element in the batch. Note that for\n        some losses, there are multiple elements per sample. If the field :attr:`size_average`\n        is set to ``False``, the losses are instead summed for each minibatch. Ignored\n        when reduce is ``False``.",
    )
    argument_parser.add_argument(
        "--ignore_index",
        type=int,
        help="Specifies a target value that is ignored\n        and does not contribute to the input gradient. When\n        :attr:`size_average` is ``True``, the loss is averaged over\n        non-ignored targets.",
        required=True,
        default=-100,
    )
    argument_parser.add_argument(
        "--reduce",
        type=bool,
        help="Deprecated (see :attr:`reduction`). By default, the\n        losses are averaged or summed over observations for each minibatch depending\n        on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per\n        batch element instead and ignores :attr:`size_average`.",
    )
    argument_parser.add_argument(
        "--reduction", type=str, help=None, required=True, default="mean"
    )
    return argument_parser


def PairwiseDistanceConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Computes the batchwise pairwise distance between vectors :math:`v_1`, :math:`v_2` using the p-norm:\n\n.. math ::\n    \\Vert x \\Vert _p = \\left( \\sum_{i=1}^n  \\vert x_i \\vert ^ p \\right) ^ {1/p}.\n\n\n    - Input1: :math:`(N, D)` where `D = vector dimension`\n    - Input2: :math:`(N, D)`, same shape as the Input1\n    - Output: :math:`(N)`. If :attr:`keepdim` is ``True``, then :math:`(N, 1)`.\nExamples::\n    >>> pdist = nn.PairwiseDistance(p=2)\n    >>> input1 = torch.randn(100, 128)\n    >>> input2 = torch.randn(100, 128)\n    >>> output = pdist(input1, input2)"
    argument_parser.add_argument(
        "--p", type=float, help="the norm degree.", required=True, default=2.0
    )
    argument_parser.add_argument(
        "--eps",
        type=float,
        help="Small value to avoid division by zero.\n       ",
        required=True,
        default=1e-06,
    )
    argument_parser.add_argument(
        "--keepdim",
        type=bool,
        help="Determines whether or not to keep the vector dimension.\n       ",
        required=True,
        default=False,
    )
    argument_parser.add_argument("--norm", type=float, help="", required=True)
    argument_parser.add_argument(
        "--__constants__",
        type=loads,
        help="",
        required=True,
        default="['norm', 'eps', 'keepdim']",
    )
    return argument_parser


def PoissonNLLLossConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Negative log likelihood loss with Poisson distribution of target.\n\nThe loss can be described as:\n\n.. math::\n    \\text{target} \\sim \\mathrm{Poisson}(\\text{input})\n\n    \\text{loss}(\\text{input}, \\text{target}) = \\text{input} - \\text{target} * \\log(\\text{input})\n                                + \\log(\\text{target!})\n\nThe last term can be omitted or approximated with Stirling formula. The\napproximation is used for target values more than 1. For targets less or\nequal to 1 zeros are added to the loss.\n\n\n        .. math::\n            \\text{target}*\\log(\\text{target}) - \\text{target} + 0.5 * \\log(2\\pi\\text{target}).\n    size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,\n        the losses are averaged over each loss element in the batch. Note that for\n        some losses, there are multiple elements per sample. If the field :attr:`size_average`\n        is set to ``False``, the losses are instead summed for each minibatch. Ignored\n        when reduce is ``False``. Default: ``True``\n    eps (float, optional): Small value to avoid evaluation of :math:`\\log(0)` when\n        :attr:`log_input = False`. Default: 1e-8\n    reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the\n        losses are averaged or summed over observations for each minibatch depending\n        on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per\n        batch element instead and ignores :attr:`size_average`. Default: ``True``\n    reduction (string, optional): Specifies the reduction to apply to the output:\n        ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,\n        ``'mean'``: the sum of the output will be divided by the number of\n        elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`\n        and :attr:`reduce` are in the process of being deprecated, and in the meantime,\n        specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``\n\nExamples::\n\n    >>> loss = nn.PoissonNLLLoss()\n    >>> log_input = torch.randn(5, 2, requires_grad=True)\n    >>> target = torch.randn(5, 2)\n    >>> output = loss(log_input, target)\n    >>> output.backward()\n\nShape:\n    - Input: :math:`(N, *)` where :math:`*` means, any number of additional\n      dimensions\n    - Target: :math:`(N, *)`, same shape as the input\n    - Output: scalar by default. If :attr:`reduction` is ``'none'``, then :math:`(N, *)`,\n      the same shape as the input"
    argument_parser.add_argument(
        "--log_input",
        type=bool,
        help="if ``True`` the loss is computed as\n        :math:`\\exp(\\text{input}) - \\text{target}*\\text{input}`, if ``False`` the loss is\n        :math:`\\text{input} - \\text{target}*\\log(\\text{input}+\\text{eps})`.",
        required=True,
        default=True,
    )
    argument_parser.add_argument(
        "--full",
        type=bool,
        help="whether to compute full loss, i. e. to add the\n        Stirling approximation term",
        required=True,
        default=False,
    )
    argument_parser.add_argument(
        "--eps", type=float, help="", required=True, default=1e-08
    )
    argument_parser.add_argument(
        "--__constants__",
        type=loads,
        help="",
        required=True,
        default="['log_input', 'full', 'eps', 'reduction']",
    )
    argument_parser.add_argument("--size_average", help=None)
    argument_parser.add_argument("--reduce", help=None)
    argument_parser.add_argument(
        "--reduction", type=str, help=None, required=True, default="mean"
    )
    return argument_parser


def SmoothL1LossConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Creates a criterion that uses a squared term if the absolute\nelement-wise error falls below beta and an L1 term otherwise.\nIt is less sensitive to outliers than the `MSELoss` and in some cases\nprevents exploding gradients (e.g. see `Fast R-CNN` paper by Ross Girshick).\nAlso known as the Huber loss:\n\n.. math::\n    \\text{loss}(x, y) = \\frac{1}{n} \\sum_{i} z_{i}\n\nwhere :math:`z_{i}` is given by:\n\n.. math::\n    z_{i} =\n    \\begin{cases}\n    0.5 (x_i - y_i)^2 / beta, & \\text{if } |x_i - y_i| < beta \\\\\n    |x_i - y_i| - 0.5 * beta, & \\text{otherwise }\n    \\end{cases}\n\n:math:`x` and :math:`y` arbitrary shapes with a total of :math:`n` elements each\nthe sum operation still operates over all the elements, and divides by :math:`n`.\n\nbeta is an optional parameter that defaults to 1.\n\nNote: When beta is set to 0, this is equivalent to :class:`L1Loss`.\nPassing a negative value in for beta will result in an exception.\n\nThe division by :math:`n` can be avoided if sets ``reduction = 'sum'``.\n\n\n    reduction (string, optional): Specifies the reduction to apply to the output:\n            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,\n            ``'mean'``: the sum of the output will be divided by the number of\n            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`\n            and :attr:`reduce` are in the process of being deprecated, and in the meantime,\n            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``\n        beta (float, optional): Specifies the threshold at which to change between L1 and L2 loss.\n            This value defaults to 1.0.\n\n\nShape:\n    - Input: :math:`(N, *)` where :math:`*` means, any number of additional\n      dimensions\n    - Target: :math:`(N, *)`, same shape as the input\n    - Output: scalar. If :attr:`reduction` is ``'none'``, then\n      :math:`(N, *)`, same shape as the input"
    argument_parser.add_argument(
        "--size_average",
        type=bool,
        help="Deprecated (see :attr:`reduction`). By default,\n        the losses are averaged over each loss element in the batch. Note that for\n        some losses, there are multiple elements per sample. If the field :attr:`size_average`\n        is set to ``False``, the losses are instead summed for each minibatch. Ignored\n        when reduce is ``False``.",
        default=True,
    )
    argument_parser.add_argument(
        "--reduce",
        type=bool,
        help="Deprecated (see :attr:`reduction`). By default, the\n        losses are averaged or summed over observations for each minibatch depending\n        on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per\n        batch element instead and ignores :attr:`size_average`.",
        default=True,
    )
    argument_parser.add_argument(
        "--__constants__",
        type=str,
        action="append",
        help="",
        required=True,
        default="reduction",
    )
    argument_parser.add_argument(
        "--reduction", type=str, help=None, required=True, default="mean"
    )
    argument_parser.add_argument(
        "--beta", type=float, help=None, required=True, default=1.0
    )
    return argument_parser


def SoftMarginLossConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Creates a criterion that optimizes a two-class classification\nlogistic loss between input tensor :math:`x` and target tensor :math:`y`\n(containing 1 or -1).\n\n.. math::\n    \\text{loss}(x, y) = \\sum_i \\frac{\\log(1 + \\exp(-y[i]*x[i]))}{\\text{x.nelement}()}\n\n\n    reduction (string, optional): Specifies the reduction to apply to the output:\n            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,\n            ``'mean'``: the sum of the output will be divided by the number of\n            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`\n            and :attr:`reduce` are in the process of being deprecated, and in the meantime,\n            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``\n\n\nShape:\n    - Input: :math:`(*)` where :math:`*` means, any number of additional\n      dimensions\n    - Target: :math:`(*)`, same shape as the input\n    - Output: scalar. If :attr:`reduction` is ``'none'``, then same shape as the input"
    argument_parser.add_argument(
        "--size_average",
        type=bool,
        help="Deprecated (see :attr:`reduction`). By default,\n        the losses are averaged over each loss element in the batch. Note that for\n        some losses, there are multiple elements per sample. If the field :attr:`size_average`\n        is set to ``False``, the losses are instead summed for each minibatch. Ignored\n        when reduce is ``False``.",
        default=True,
    )
    argument_parser.add_argument(
        "--reduce",
        type=bool,
        help="Deprecated (see :attr:`reduction`). By default, the\n        losses are averaged or summed over observations for each minibatch depending\n        on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per\n        batch element instead and ignores :attr:`size_average`.",
        default=True,
    )
    argument_parser.add_argument(
        "--__constants__",
        type=str,
        action="append",
        help="",
        required=True,
        default="reduction",
    )
    argument_parser.add_argument(
        "--reduction", type=str, help=None, required=True, default="mean"
    )
    return argument_parser


def TripletMarginLossConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Creates a criterion that measures the triplet loss given an input\ntensors :math:`x1`, :math:`x2`, :math:`x3` and a margin with a value greater than :math:`0`.\nThis is used for measuring a relative similarity between samples. A triplet\nis composed by `a`, `p` and `n` (i.e., `anchor`, `positive examples` and `negative\nexamples` respectively). The shapes of all input tensors should be\n:math:`(N, D)`.\n\nThe distance swap is described in detail in the paper `Learning shallow\nconvolutional feature descriptors with triplet losses`_ by\nV. Balntas, E. Riba et al.\n\nThe loss function for each sample in the mini-batch is:\n\n.. math::\n    L(a, p, n) = \\max \\{d(a_i, p_i) - d(a_i, n_i) + {\\rm margin}, 0\\}\n\n\nwhere\n\n.. math::\n    d(x_i, y_i) = \\left\\lVert {\\bf x}_i - {\\bf y}_i \\right\\rVert_p\n\nSee also :class:`~torch.nn.TripletMarginWithDistanceLoss`, which computes the\ntriplet margin loss for input tensors using a custom distance function.\n\n\n    reduction (string, optional): Specifies the reduction to apply to the output:\n            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,\n            ``'mean'``: the sum of the output will be divided by the number of\n            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`\n            and :attr:`reduce` are in the process of being deprecated, and in the meantime,\n            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``\n\n\nShape:\n    - Input: :math:`(N, D)` where :math:`D` is the vector dimension.\n    - Output: A Tensor of shape :math:`(N)` if :attr:`reduction` is ``'none'``, or a scalar\n        otherwise.\n\n>>> triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)\n>>> anchor = torch.randn(100, 128, requires_grad=True)\n>>> positive = torch.randn(100, 128, requires_grad=True)\n>>> negative = torch.randn(100, 128, requires_grad=True)\n>>> output = triplet_loss(anchor, positive, negative)\n>>> output.backward()\n\n.. _Learning shallow convolutional feature descriptors with triplet losses:\n    http://www.bmva.org/bmvc/2016/papers/paper119/index.html"
    argument_parser.add_argument(
        "--margin", type=float, help="Default: :math:`1`", required=True, default=1.0
    )
    argument_parser.add_argument(
        "--p",
        type=float,
        help="The norm degree for pairwise distance.",
        required=True,
        default=2.0,
    )
    argument_parser.add_argument(
        "--swap",
        type=bool,
        help="The distance swap is described in detail in the paper\n        `Learning shallow convolutional feature descriptors with triplet losses` by\n        V. Balntas, E. Riba et al.",
        required=True,
        default=False,
    )
    argument_parser.add_argument(
        "--size_average",
        type=bool,
        help="Deprecated (see :attr:`reduction`). By default,\n        the losses are averaged over each loss element in the batch. Note that for\n        some losses, there are multiple elements per sample. If the field :attr:`size_average`\n        is set to ``False``, the losses are instead summed for each minibatch. Ignored\n        when reduce is ``False``.",
        default=True,
    )
    argument_parser.add_argument(
        "--reduce",
        type=bool,
        help="Deprecated (see :attr:`reduction`). By default, the\n        losses are averaged or summed over observations for each minibatch depending\n        on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per\n        batch element instead and ignores :attr:`size_average`.",
        default=True,
    )
    argument_parser.add_argument(
        "--eps", type=float, help="", required=True, default=1e-06
    )
    argument_parser.add_argument(
        "--__constants__",
        type=loads,
        help="",
        required=True,
        default="['margin', 'p', 'eps', 'swap', 'reduction']",
    )
    argument_parser.add_argument(
        "--reduction", type=str, help=None, required=True, default="mean"
    )
    return argument_parser


def TripletMarginWithDistanceLossConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Creates a criterion that measures the triplet loss given input\ntensors :math:`a`, :math:`p`, and :math:`n` (representing anchor,\npositive, and negative examples, respectively), and a nonnegative,\nreal-valued function (\"distance function\") used to compute the relationship\nbetween the anchor and positive example (\"positive distance\") and the\nanchor and negative example (\"negative distance\").\n\nThe unreduced loss (i.e., with :attr:`reduction` set to ``'none'``)\ncan be described as:\n\n.. math::\n    \\ell(a, p, n) = L = \\{l_1,\\dots,l_N\\}^\\top, \\quad\n    l_i = \\max \\{d(a_i, p_i) - d(a_i, n_i) + {\\rm margin}, 0\\}\n\nwhere :math:`N` is the batch size; :math:`d` is a nonnegative, real-valued function\nquantifying the closeness of two tensors, referred to as the :attr:`distance_function`;\nand :math:`margin` is a non-negative margin representing the minimum difference\nbetween the positive and negative distances that is required for the loss to\nbe 0.  The input tensors have :math:`N` elements each and can be of any shape\nthat the distance function can handle.\n\nIf :attr:`reduction` is not ``'none'``\n(default ``'mean'``), then:\n\n.. math::\n    \\ell(x, y) =\n    \\begin{cases}\n        \\operatorname{mean}(L), &  \\text{if reduction} = \\text{`mean';}\\\\\n        \\operatorname{sum}(L),  &  \\text{if reduction} = \\text{`sum'.}\n    \\end{cases}\n\nSee also :class:`~torch.nn.TripletMarginLoss`, which computes the triplet\nloss for input tensors using the :math:`l_p` distance as the distance function.\n\n\n    reduction (string, optional): Specifies the (optional) reduction to apply to the output:\n            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,\n            ``'mean'``: the sum of the output will be divided by the number of\n            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``\n\n\n\nShape:\n    - Input: :math:`(N, *)` where :math:`*` represents any number of additional dimensions\n      as supported by the distance function.\n    - Output: A Tensor of shape :math:`(N)` if :attr:`reduction` is ``'none'``, or a scalar\n      otherwise.\n\nExamples::\n\n>>> # Initialize embeddings\n>>> embedding = nn.Embedding(1000, 128)\n>>> anchor_ids = torch.randint(0, 1000, (1,), requires_grad=True)\n>>> positive_ids = torch.randint(0, 1000, (1,), requires_grad=True)\n>>> negative_ids = torch.randint(0, 1000, (1,), requires_grad=True)\n>>> anchor = embedding(anchor_ids)\n>>> positive = embedding(positive_ids)\n>>> negative = embedding(negative_ids)\n>>>\n>>> # Built-in Distance Function\n>>> triplet_loss = \\\n>>>     nn.TripletMarginWithDistanceLoss(distance_function=nn.PairwiseDistance())\n>>> output = triplet_loss(anchor, positive, negative)\n>>> output.backward()\n>>>\n>>> # Custom Distance Function\n>>> def l_infinity(x1, x2):\n>>>     return torch.max(torch.abs(x1 - x2), dim=1).values\n>>>\n>>> triplet_loss = \\\n>>>     nn.TripletMarginWithDistanceLoss(distance_function=l_infinity, margin=1.5)\n>>> output = triplet_loss(anchor, positive, negative)\n>>> output.backward()\n>>>\n>>> # Custom Distance Function (Lambda)\n>>> triplet_loss = \\\n>>>     nn.TripletMarginWithDistanceLoss(\n>>>         distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y))\n>>> output = triplet_loss(anchor, positive, negative)\n>>> output.backward()\n\nReference:\n    V. Balntas, et al.: Learning shallow convolutional feature descriptors with triplet losses:\n    http://www.bmva.org/bmvc/2016/papers/paper119/index.html"
    argument_parser.add_argument(
        "--distance_function",
        type=str,
        help="A nonnegative, real-valued function that\n        quantifies the closeness of two tensors. If not specified,\n        `nn.PairwiseDistance` will be used. ",
        default="None",
    )
    argument_parser.add_argument(
        "--margin",
        type=float,
        help="A non-negative margin representing the minimum difference\n        between the positive and negative distances required for the loss to be 0. Larger\n        margins penalize cases where the negative examples are not distant enough from the\n        anchors, relative to the positives.",
        required=True,
        default=1.0,
    )
    argument_parser.add_argument(
        "--swap",
        type=bool,
        help="Whether to use the distance swap described in the paper\n        `Learning shallow convolutional feature descriptors with triplet losses` by\n        V. Balntas, E. Riba et al. If True, and if the positive example is closer to the\n        negative example than the anchor is, swaps the positive example and the anchor in\n        the loss computation.",
        required=True,
        default=False,
    )
    argument_parser.add_argument(
        "--__constants__",
        type=loads,
        help="",
        required=True,
        default="['margin', 'swap', 'reduction']",
    )
    return argument_parser


__all__ = [
    "BCELossConfig",
    "BCEWithLogitsLossConfig",
    "CTCLossConfig",
    "CosineEmbeddingLossConfig",
    "CrossEntropyLossConfig",
    "HingeEmbeddingLossConfig",
    "KLDivLossConfig",
    "L1LossConfig",
    "MSELossConfig",
    "MarginRankingLossConfig",
    "MultiLabelMarginLossConfig",
    "MultiLabelSoftMarginLossConfig",
    "MultiMarginLossConfig",
    "NLLLossConfig",
    "NLLLoss2dConfig",
    "PairwiseDistanceConfig",
    "PoissonNLLLossConfig",
    "SmoothL1LossConfig",
    "SoftMarginLossConfig",
    "TripletMarginLossConfig",
    "TripletMarginWithDistanceLossConfig",
]
