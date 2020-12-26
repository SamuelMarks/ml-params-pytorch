""" Generated Activation CLI parsers """
from yaml import safe_load as loads


def CELUConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Applies the element-wise function:\n\n.. math::\n    \\text{CELU}(x) = \\max(0,x) + \\min(0, \\alpha * (\\exp(x/\\alpha) - 1))\n\nMore details can be found in the paper `Continuously Differentiable Exponential Linear Units`_ .\n\n\nShape:\n    - Input: :math:`(N, *)` where `*` means, any number of additional\n      dimensions\n    - Output: :math:`(N, *)`, same shape as the input\n\n.. image:: ../scripts/activation_images/CELU.png\n\nExamples::\n\n    >>> m = nn.CELU()\n    >>> input = torch.randn(2)\n    >>> output = m(input)\n\n.. _`Continuously Differentiable Exponential Linear Units`:\n    https://arxiv.org/abs/1704.07483"
    argument_parser.add_argument(
        "--alpha",
        type=float,
        help="the :math:`\\alpha` value for the CELU formulation.",
        required=True,
        default=1.0,
    )
    argument_parser.add_argument(
        "--inplace",
        type=bool,
        help="can optionally do the operation in-place.",
        required=True,
        default=False,
    )
    argument_parser.add_argument(
        "--__constants__",
        type=loads,
        help="",
        required=True,
        default="['alpha', 'inplace']",
    )
    return argument_parser


def ELUConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Applies the element-wise function:\n\n.. math::\n    \\text{ELU}(x) = \\begin{cases}\n    x, & \\text{ if } x > 0\\\\\n    \\alpha * (\\exp(x) - 1), & \\text{ if } x \\leq 0\n    \\end{cases}\n\n\nShape:\n    - Input: :math:`(N, *)` where `*` means, any number of additional\n      dimensions\n    - Output: :math:`(N, *)`, same shape as the input\n\n.. image:: ../scripts/activation_images/ELU.png\n\nExamples::\n\n    >>> m = nn.ELU()\n    >>> input = torch.randn(2)\n    >>> output = m(input)"
    argument_parser.add_argument(
        "--alpha",
        type=float,
        help="the :math:`\\alpha` value for the ELU formulation.",
        required=True,
        default=1.0,
    )
    argument_parser.add_argument(
        "--inplace",
        type=bool,
        help="can optionally do the operation in-place.",
        required=True,
        default=False,
    )
    argument_parser.add_argument(
        "--__constants__",
        type=loads,
        help="",
        required=True,
        default="['alpha', 'inplace']",
    )
    return argument_parser


def GELUConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Applies the Gaussian Error Linear Units function:\n\n.. math:: \\text{GELU}(x) = x * \\Phi(x)\n\nwhere :math:`\\Phi(x)` is the Cumulative Distribution Function for Gaussian Distribution.\n\nShape:\n    - Input: :math:`(N, *)` where `*` means, any number of additional\n      dimensions\n    - Output: :math:`(N, *)`, same shape as the input\n\n.. image:: ../scripts/activation_images/GELU.png\n\nExamples::\n\n    >>> m = nn.GELU()\n    >>> input = torch.randn(2)\n    >>> output = m(input)"
    return argument_parser


def GLUConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Applies the gated linear unit function\n:math:`{GLU}(a, b)= a \\otimes \\sigma(b)` where :math:`a` is the first half\nof the input matrices and :math:`b` is the second half.\n\n\nShape:\n    - Input: :math:`(\\ast_1, N, \\ast_2)` where `*` means, any number of additional\n      dimensions\n    - Output: :math:`(\\ast_1, M, \\ast_2)` where :math:`M=N/2`\n\nExamples::\n\n    >>> m = nn.GLU()\n    >>> input = torch.randn(4, 2)\n    >>> output = m(input)"
    argument_parser.add_argument(
        "--dim",
        type=int,
        help="the dimension on which to split the input.",
        required=True,
        default=-1,
    )
    argument_parser.add_argument(
        "--__constants__",
        type=str,
        action="append",
        help="",
        required=True,
        default="dim",
    )
    return argument_parser


def HardshrinkConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Applies the hard shrinkage function element-wise:\n\n.. math::\n    \\text{HardShrink}(x) =\n    \\begin{cases}\n    x, & \\text{ if } x > \\lambda \\\\\n    x, & \\text{ if } x < -\\lambda \\\\\n    0, & \\text{ otherwise }\n    \\end{cases}\n\n\nShape:\n    - Input: :math:`(N, *)` where `*` means, any number of additional\n      dimensions\n    - Output: :math:`(N, *)`, same shape as the input\n\n.. image:: ../scripts/activation_images/Hardshrink.png\n\nExamples::\n\n    >>> m = nn.Hardshrink()\n    >>> input = torch.randn(2)\n    >>> output = m(input)"
    argument_parser.add_argument(
        "--lambd",
        type=float,
        help="the :math:`\\lambda` value for the Hardshrink formulation.",
        required=True,
        default=0.5,
    )
    argument_parser.add_argument(
        "--__constants__",
        type=str,
        action="append",
        help="",
        required=True,
        default="lambd",
    )
    return argument_parser


def HardsigmoidConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Applies the element-wise function:\n\n.. math::\n    \\text{Hardsigmoid}(x) = \\begin{cases}\n        0 & \\text{if~} x \\le -3, \\\\\n        1 & \\text{if~} x \\ge +3, \\\\\n        x / 6 + 1 / 2 & \\text{otherwise}\n    \\end{cases}\n\n\nShape:\n    - Input: :math:`(N, *)` where `*` means, any number of additional\n      dimensions\n    - Output: :math:`(N, *)`, same shape as the input\n\nExamples::\n\n    >>> m = nn.Hardsigmoid()\n    >>> input = torch.randn(2)\n    >>> output = m(input)"
    argument_parser.add_argument(
        "--inplace",
        type=bool,
        help="can optionally do the operation in-place.",
        required=True,
        default=False,
    )
    argument_parser.add_argument(
        "--__constants__",
        type=str,
        action="append",
        help="",
        required=True,
        default="inplace",
    )
    return argument_parser


def HardswishConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Applies the hardswish function, element-wise, as described in the paper:\n\n`Searching for MobileNetV3`_.\n\n.. math::\n    \\text{Hardswish}(x) = \\begin{cases}\n        0 & \\text{if~} x \\le -3, \\\\\n        x & \\text{if~} x \\ge +3, \\\\\n        x \\cdot (x + 3) /6 & \\text{otherwise}\n    \\end{cases}\n\n\nShape:\n    - Input: :math:`(N, *)` where `*` means, any number of additional\n      dimensions\n    - Output: :math:`(N, *)`, same shape as the input\n\nExamples::\n\n    >>> m = nn.Hardswish()\n    >>> input = torch.randn(2)\n    >>> output = m(input)\n\n.. _`Searching for MobileNetV3`:\n    https://arxiv.org/abs/1905.02244"
    argument_parser.add_argument(
        "--inplace",
        type=bool,
        help="can optionally do the operation in-place.",
        required=True,
        default=False,
    )
    argument_parser.add_argument(
        "--__constants__",
        type=str,
        action="append",
        help="",
        required=True,
        default="inplace",
    )
    return argument_parser


def HardtanhConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Applies the HardTanh function element-wise\n\nHardTanh is defined as:\n\n.. math::\n    \\text{HardTanh}(x) = \\begin{cases}\n        1 & \\text{ if } x > 1 \\\\\n        -1 & \\text{ if } x < -1 \\\\\n        x & \\text{ otherwise } \\\\\n    \\end{cases}\n\nThe range of the linear region :math:`[-1, 1]` can be adjusted using\n:attr:`min_val` and :attr:`max_val`.\n\n\nKeyword arguments :attr:`min_value` and :attr:`max_value`\nhave been deprecated in favor of :attr:`min_val` and :attr:`max_val`.\n\nShape:\n    - Input: :math:`(N, *)` where `*` means, any number of additional\n      dimensions\n    - Output: :math:`(N, *)`, same shape as the input\n\n.. image:: ../scripts/activation_images/Hardtanh.png\n\nExamples::\n\n    >>> m = nn.Hardtanh(-2, 2)\n    >>> input = torch.randn(2)\n    >>> output = m(input)"
    argument_parser.add_argument(
        "--min_val",
        type=float,
        help="minimum value of the linear region range.",
        required=True,
        default=-1.0,
    )
    argument_parser.add_argument(
        "--max_val",
        type=float,
        help="maximum value of the linear region range.",
        required=True,
        default=1.0,
    )
    argument_parser.add_argument(
        "--inplace",
        type=bool,
        help="can optionally do the operation in-place.",
        required=True,
        default=False,
    )
    argument_parser.add_argument(
        "--__constants__",
        type=loads,
        help="",
        required=True,
        default="['min_val', 'max_val', 'inplace']",
    )
    argument_parser.add_argument("--min_value", type=float, help=None)
    argument_parser.add_argument("--max_value", type=float, help=None)
    return argument_parser


def LeakyReLUConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Applies the element-wise function:\n\n.. math::\n    \\text{LeakyReLU}(x) = \\max(0, x) + \\text{negative\\_slope} * \\min(0, x)\n\n\nor\n\n.. math::\n    \\text{LeakyRELU}(x) =\n    \\begin{cases}\n    x, & \\text{ if } x \\geq 0 \\\\\n    \\text{negative\\_slope} \\times x, & \\text{ otherwise }\n    \\end{cases}\n\n\nShape:\n    - Input: :math:`(N, *)` where `*` means, any number of additional\n      dimensions\n    - Output: :math:`(N, *)`, same shape as the input\n\n.. image:: ../scripts/activation_images/LeakyReLU.png\n\nExamples::\n\n    >>> m = nn.LeakyReLU(0.1)\n    >>> input = torch.randn(2)\n    >>> output = m(input)"
    argument_parser.add_argument(
        "--negative_slope",
        type=float,
        help="Controls the angle of the negative slope.",
        required=True,
        default=0.01,
    )
    argument_parser.add_argument(
        "--inplace",
        type=bool,
        help="can optionally do the operation in-place.",
        required=True,
        default=False,
    )
    argument_parser.add_argument(
        "--__constants__",
        type=loads,
        help="",
        required=True,
        default="['inplace', 'negative_slope']",
    )
    return argument_parser


def LogSigmoidConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Applies the element-wise function:\n\n.. math::\n    \\text{LogSigmoid}(x) = \\log\\left(\\frac{ 1 }{ 1 + \\exp(-x)}\\right)\n\nShape:\n    - Input: :math:`(N, *)` where `*` means, any number of additional\n      dimensions\n    - Output: :math:`(N, *)`, same shape as the input\n\n.. image:: ../scripts/activation_images/LogSigmoid.png\n\nExamples::\n\n    >>> m = nn.LogSigmoid()\n    >>> input = torch.randn(2)\n    >>> output = m(input)"
    return argument_parser


def LogSoftmaxConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser, values in the range [-inf, 0)
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Applies the :math:`\\log(\\text{Softmax}(x))` function to an n-dimensional\ninput Tensor. The LogSoftmax formulation can be simplified as:\n\n.. math::\n    \\text{LogSoftmax}(x_{i}) = \\log\\left(\\frac{\\exp(x_i) }{ \\sum_j \\exp(x_j)} \\right)\n\nShape:\n    - Input: :math:`(*)` where `*` means, any number of additional\n      dimensions\n    - Output: :math:`(*)`, same shape as the input\n\nArguments:\n    dim (int): A dimension along which LogSoftmax will be computed.\n\n\nExamples::\n\n    >>> m = nn.LogSoftmax()\n    >>> input = torch.randn(2, 3)\n    >>> output = m(input)"
    argument_parser.add_argument(
        "--__constants__",
        type=str,
        action="append",
        help="",
        required=True,
        default="dim",
    )
    argument_parser.add_argument("--dim", type=int, help="")
    return argument_parser


def MultiheadAttentionConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Allows the model to jointly attend to information\nfrom different representation subspaces.\nSee reference: Attention Is All You Need\n\n.. math::\n    \\text{MultiHead}(Q, K, V) = \\text{Concat}(head_1,\\dots,head_h)W^O\n    \\text{where} head_i = \\text{Attention}(QW_i^Q, KW_i^K, VW_i^V)\n\n\n    Note: if kdim and vdim are None, they will be set to embed_dim such that\n    query, key, and value have the same number of features.\n\nExamples::\n\n    >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)\n    >>> attn_output, attn_output_weights = multihead_attn(query, key, value)"
    argument_parser.add_argument(
        "--embed_dim",
        type=float,
        help="total dimension of the model.",
        required=True,
        default=0.0,
    )
    argument_parser.add_argument(
        "--num_heads",
        type=bool,
        help="parallel attention heads.",
        required=True,
        default=True,
    )
    argument_parser.add_argument(
        "--dropout",
        type=float,
        help="a Dropout layer on attn_output_weights.",
        required=True,
        default=0.0,
    )
    argument_parser.add_argument(
        "--bias",
        type=bool,
        help="add bias as module parameter.",
        required=True,
        default=True,
    )
    argument_parser.add_argument(
        "--add_bias_kv",
        type=bool,
        help="add bias to the key and value sequences at dim=0.",
        required=True,
        default=False,
    )
    argument_parser.add_argument(
        "--add_zero_attn",
        type=bool,
        help="add a new batch of zeros to the key and\n                   value sequences at dim=1.",
        required=True,
        default=False,
    )
    argument_parser.add_argument(
        "--kdim",
        type=str,
        help="total number of features in key.",
        required=True,
        default="None",
    )
    argument_parser.add_argument(
        "--vdim",
        type=str,
        help="total number of features in value.",
        required=True,
        default="None",
    )
    argument_parser.add_argument("--bias_v", type=str, help="")
    argument_parser.add_argument("--bias_k", type=str, help="")
    return argument_parser


def PReLUConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Applies the element-wise function:\n\n.. math::\n    \\text{PReLU}(x) = \\max(0,x) + a * \\min(0,x)\n\nor\n\n.. math::\n    \\text{PReLU}(x) =\n    \\begin{cases}\n    x, & \\text{ if } x \\geq 0 \\\\\n    ax, & \\text{ otherwise }\n    \\end{cases}\n\nHere :math:`a` is a learnable parameter. When called without arguments, `nn.PReLU()` uses a single\nparameter :math:`a` across all input channels. If called with `nn.PReLU(nChannels)`,\na separate :math:`a` is used for each input channel.\n\n\n.. note::\n    weight decay should not be used when learning :math:`a` for good performance.\n\n.. note::\n    Channel dim is the 2nd dim of input. When input has dims < 2, then there is\n    no channel dim and the number of channels = 1.\n\n\nShape:\n    - Input: :math:`(N, *)` where `*` means, any number of additional\n      dimensions\n    - Output: :math:`(N, *)`, same shape as the input\n\nAttributes:\n    weight (Tensor): the learnable weights of shape (:attr:`num_parameters`).\n\n.. image:: ../scripts/activation_images/PReLU.png\n\nExamples::\n\n    >>> m = nn.PReLU()\n    >>> input = torch.randn(2)\n    >>> output = m(input)"
    argument_parser.add_argument(
        "--num_parameters",
        type=int,
        help="number of :math:`a` to learn.\n        Although it takes an int as input, there is only two values are legitimate:\n        1, or the number of channels at input.",
        required=True,
        default=1,
    )
    argument_parser.add_argument(
        "--init",
        type=float,
        help="the initial value of :math:`a`.",
        required=True,
        default=0.25,
    )
    argument_parser.add_argument(
        "--__constants__",
        type=str,
        action="append",
        help="",
        required=True,
        default="num_parameters",
    )
    return argument_parser


def RReLUConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Applies the randomized leaky rectified liner unit function, element-wise,\nas described in the paper:\n\n`Empirical Evaluation of Rectified Activations in Convolutional Network`_.\n\nThe function is defined as:\n\n.. math::\n    \\text{RReLU}(x) =\n    \\begin{cases}\n        x & \\text{if } x \\geq 0 \\\\\n        ax & \\text{ otherwise }\n    \\end{cases}\n\nwhere :math:`a` is randomly sampled from uniform distribution\n:math:`\\mathcal{U}(\\text{lower}, \\text{upper})`.\n\n See: https://arxiv.org/pdf/1505.00853.pdf\n\n\nShape:\n    - Input: :math:`(N, *)` where `*` means, any number of additional\n      dimensions\n    - Output: :math:`(N, *)`, same shape as the input\n\nExamples::\n\n    >>> m = nn.RReLU(0.1, 0.3)\n    >>> input = torch.randn(2)\n    >>> output = m(input)\n\n.. _`Empirical Evaluation of Rectified Activations in Convolutional Network`:\n    https://arxiv.org/abs/1505.00853"
    argument_parser.add_argument(
        "--lower",
        type=float,
        help="lower bound of the uniform distribution.",
        required=True,
        default=0.125,
    )
    argument_parser.add_argument(
        "--upper",
        type=float,
        help="upper bound of the uniform distribution.",
        required=True,
        default=0.3333333333333333,
    )
    argument_parser.add_argument(
        "--inplace",
        type=bool,
        help="can optionally do the operation in-place.",
        required=True,
        default=False,
    )
    argument_parser.add_argument(
        "--__constants__",
        type=loads,
        help="",
        required=True,
        default="['lower', 'upper', 'inplace']",
    )
    return argument_parser


def ReLUConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Applies the rectified linear unit function element-wise:\n\n:math:`\\text{ReLU}(x) = (x)^+ = \\max(0, x)`\n\n\nShape:\n    - Input: :math:`(N, *)` where `*` means, any number of additional\n      dimensions\n    - Output: :math:`(N, *)`, same shape as the input\n\n.. image:: ../scripts/activation_images/ReLU.png\n\nExamples::\n\n    >>> m = nn.ReLU()\n    >>> input = torch.randn(2)\n    >>> output = m(input)\n\n\n  An implementation of CReLU - https://arxiv.org/abs/1603.05201\n\n    >>> m = nn.ReLU()\n    >>> input = torch.randn(2).unsqueeze(0)\n    >>> output = torch.cat((m(input),m(-input)))"
    argument_parser.add_argument(
        "--inplace",
        type=bool,
        help="can optionally do the operation in-place.",
        required=True,
        default=False,
    )
    argument_parser.add_argument(
        "--__constants__",
        type=str,
        action="append",
        help="",
        required=True,
        default="inplace",
    )
    return argument_parser


def ReLU6Config(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Applies the element-wise function:\n\n.. math::\n    \\text{ReLU6}(x) = \\min(\\max(0,x), 6)\n\n\nShape:\n    - Input: :math:`(N, *)` where `*` means, any number of additional\n      dimensions\n    - Output: :math:`(N, *)`, same shape as the input\n\n.. image:: ../scripts/activation_images/ReLU6.png\n\nExamples::\n\n    >>> m = nn.ReLU6()\n    >>> input = torch.randn(2)\n    >>> output = m(input)"
    argument_parser.add_argument(
        "--inplace",
        type=bool,
        help="can optionally do the operation in-place.",
        required=True,
        default=False,
    )
    return argument_parser


def SELUConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Applied element-wise, as:\n\n.. math::\n    \\text{SELU}(x) = \\text{scale} * (\\max(0,x) + \\min(0, \\alpha * (\\exp(x) - 1)))\n\nwith :math:`\\alpha = 1.6732632423543772848170429916717` and\n:math:`\\text{scale} = 1.0507009873554804934193349852946`.\n\nMore details can be found in the paper `Self-Normalizing Neural Networks`_ .\n\n\nShape:\n    - Input: :math:`(N, *)` where `*` means, any number of additional\n      dimensions\n    - Output: :math:`(N, *)`, same shape as the input\n\n.. image:: ../scripts/activation_images/SELU.png\n\nExamples::\n\n    >>> m = nn.SELU()\n    >>> input = torch.randn(2)\n    >>> output = m(input)\n\n.. _Self-Normalizing Neural Networks: https://arxiv.org/abs/1706.02515"
    argument_parser.add_argument(
        "--inplace",
        type=bool,
        help="can optionally do the operation in-place.",
        required=True,
        default=False,
    )
    argument_parser.add_argument(
        "--__constants__",
        type=str,
        action="append",
        help="",
        required=True,
        default="inplace",
    )
    return argument_parser


def SiLUConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Applies the silu function, element-wise.\n\n.. math::\n    \\text{silu}(x) = x * \\sigma(x), \\text{where } \\sigma(x) \\text{ is the logistic sigmoid.}\n\n.. note::\n    See `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_ \n    where the SiLU (Sigmoid Linear Unit) was originally coined, and see \n    `Sigmoid-Weighted Linear Units for Neural Network Function Approximation \n    in Reinforcement Learning <https://arxiv.org/abs/1702.03118>`_ and `Swish: \n    a Self-Gated Activation Function <https://arxiv.org/abs/1710.05941v1>`_ \n    where the SiLU was experimented with later.\n\nShape:\n    - Input: :math:`(N, *)` where `*` means, any number of additional\n      dimensions\n    - Output: :math:`(N, *)`, same shape as the input\n\nExamples::\n\n    >>> m = nn.SiLU()\n    >>> input = torch.randn(2)\n    >>> output = m(input)"
    argument_parser.add_argument(
        "--__constants__",
        type=str,
        action="append",
        help="",
        required=True,
        default="inplace",
    )
    argument_parser.add_argument(
        "--inplace", type=bool, help="", required=True, default=False
    )
    return argument_parser


def SigmoidConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Applies the element-wise function:\n\n.. math::\n    \\text{Sigmoid}(x) = \\sigma(x) = \\frac{1}{1 + \\exp(-x)}\n\n\nShape:\n    - Input: :math:`(N, *)` where `*` means, any number of additional\n      dimensions\n    - Output: :math:`(N, *)`, same shape as the input\n\n.. image:: ../scripts/activation_images/Sigmoid.png\n\nExamples::\n\n    >>> m = nn.Sigmoid()\n    >>> input = torch.randn(2)\n    >>> output = m(input)"
    return argument_parser


def SoftmaxConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser, values in the range [0, 1]
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Applies the Softmax function to an n-dimensional input Tensor\nrescaling them so that the elements of the n-dimensional output Tensor\nlie in the range [0,1] and sum to 1.\n\nSoftmax is defined as:\n\n.. math::\n    \\text{Softmax}(x_{i}) = \\frac{\\exp(x_i)}{\\sum_j \\exp(x_j)}\n\nWhen the input Tensor is a sparse tensor then the unspecifed\nvalues are treated as ``-inf``.\n\nShape:\n    - Input: :math:`(*)` where `*` means, any number of additional\n      dimensions\n    - Output: :math:`(*)`, same shape as the input\n\n\nArguments:\n    dim (int): A dimension along which Softmax will be computed (so every slice\n        along dim will sum to 1).\n\n.. note::\n    This module doesn't work directly with NLLLoss,\n    which expects the Log to be computed between the Softmax and itself.\n    Use `LogSoftmax` instead (it's faster and has better numerical properties).\n\nExamples::\n\n    >>> m = nn.Softmax(dim=1)\n    >>> input = torch.randn(2, 3)\n    >>> output = m(input)"
    argument_parser.add_argument(
        "--__constants__",
        type=str,
        action="append",
        help="",
        required=True,
        default="dim",
    )
    argument_parser.add_argument("--dim", type=int, help="")
    return argument_parser


def Softmax2dConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser, values in the range [0, 1]
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Applies SoftMax over features to each spatial location.\n\nWhen given an image of ``Channels x Height x Width``, it will\napply `Softmax` to each location :math:`(Channels, h_i, w_j)`\n\nShape:\n    - Input: :math:`(N, C, H, W)`\n    - Output: :math:`(N, C, H, W)` (same shape as input)\n\n\nExamples::\n\n    >>> m = nn.Softmax2d()\n    >>> # you softmax over the 2nd dimension\n    >>> input = torch.randn(2, 3, 12, 13)\n    >>> output = m(input)"
    return argument_parser


def SoftminConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser, values in the range [0, 1]
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Applies the Softmin function to an n-dimensional input Tensor\nrescaling them so that the elements of the n-dimensional output Tensor\nlie in the range `[0, 1]` and sum to 1.\n\nSoftmin is defined as:\n\n.. math::\n    \\text{Softmin}(x_{i}) = \\frac{\\exp(-x_i)}{\\sum_j \\exp(-x_j)}\n\nShape:\n    - Input: :math:`(*)` where `*` means, any number of additional\n      dimensions\n    - Output: :math:`(*)`, same shape as the input\n\nArguments:\n    dim (int): A dimension along which Softmin will be computed (so every slice\n        along dim will sum to 1).\n\n\nExamples::\n\n    >>> m = nn.Softmin()\n    >>> input = torch.randn(2, 3)\n    >>> output = m(input)"
    argument_parser.add_argument(
        "--__constants__",
        type=str,
        action="append",
        help="",
        required=True,
        default="dim",
    )
    argument_parser.add_argument("--dim", type=int, help="")
    return argument_parser


def SoftplusConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Applies the element-wise function:\n\n.. math::\n    \\text{Softplus}(x) = \\frac{1}{\\beta} * \\log(1 + \\exp(\\beta * x))\n\nSoftPlus is a smooth approximation to the ReLU function and can be used\nto constrain the output of a machine to always be positive.\n\nFor numerical stability the implementation reverts to the linear function\nwhen :math:`input \\times \\beta > threshold`.\n\n\nShape:\n    - Input: :math:`(N, *)` where `*` means, any number of additional\n      dimensions\n    - Output: :math:`(N, *)`, same shape as the input\n\n.. image:: ../scripts/activation_images/Softplus.png\n\nExamples::\n\n    >>> m = nn.Softplus()\n    >>> input = torch.randn(2)\n    >>> output = m(input)"
    argument_parser.add_argument(
        "--beta",
        type=int,
        help="the :math:`\\beta` value for the Softplus formulation.",
        required=True,
        default=1,
    )
    argument_parser.add_argument(
        "--threshold",
        type=int,
        help="values above this revert to a linear function.",
        required=True,
        default=20,
    )
    argument_parser.add_argument(
        "--__constants__",
        type=loads,
        help="",
        required=True,
        default="['beta', 'threshold']",
    )
    return argument_parser


def SoftshrinkConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Applies the soft shrinkage function elementwise:\n\n.. math::\n    \\text{SoftShrinkage}(x) =\n    \\begin{cases}\n    x - \\lambda, & \\text{ if } x > \\lambda \\\\\n    x + \\lambda, & \\text{ if } x < -\\lambda \\\\\n    0, & \\text{ otherwise }\n    \\end{cases}\n\n\nShape:\n    - Input: :math:`(N, *)` where `*` means, any number of additional\n      dimensions\n    - Output: :math:`(N, *)`, same shape as the input\n\n.. image:: ../scripts/activation_images/Softshrink.png\n\nExamples::\n\n    >>> m = nn.Softshrink()\n    >>> input = torch.randn(2)\n    >>> output = m(input)"
    argument_parser.add_argument(
        "--lambd",
        type=float,
        help="the :math:`\\lambda` (must be no less than zero) value for the Softshrink formulation.",
        required=True,
        default=0.5,
    )
    argument_parser.add_argument(
        "--__constants__",
        type=str,
        action="append",
        help="",
        required=True,
        default="lambd",
    )
    return argument_parser


def SoftsignConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Applies the element-wise function:\n\n.. math::\n    \\text{SoftSign}(x) = \\frac{x}{ 1 + |x|}\n\nShape:\n    - Input: :math:`(N, *)` where `*` means, any number of additional\n      dimensions\n    - Output: :math:`(N, *)`, same shape as the input\n\n.. image:: ../scripts/activation_images/Softsign.png\n\nExamples::\n\n    >>> m = nn.Softsign()\n    >>> input = torch.randn(2)\n    >>> output = m(input)"
    return argument_parser


def TanhConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Applies the element-wise function:\n\n.. math::\n    \\text{Tanh}(x) = \\tanh(x) = \\frac{\\exp(x) - \\exp(-x)} {\\exp(x) + \\exp(-x)}\n\nShape:\n    - Input: :math:`(N, *)` where `*` means, any number of additional\n      dimensions\n    - Output: :math:`(N, *)`, same shape as the input\n\n.. image:: ../scripts/activation_images/Tanh.png\n\nExamples::\n\n    >>> m = nn.Tanh()\n    >>> input = torch.randn(2)\n    >>> output = m(input)"
    return argument_parser


def TanhshrinkConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Applies the element-wise function:\n\n.. math::\n    \\text{Tanhshrink}(x) = x - \\tanh(x)\n\nShape:\n    - Input: :math:`(N, *)` where `*` means, any number of additional\n      dimensions\n    - Output: :math:`(N, *)`, same shape as the input\n\n.. image:: ../scripts/activation_images/Tanhshrink.png\n\nExamples::\n\n    >>> m = nn.Tanhshrink()\n    >>> input = torch.randn(2)\n    >>> output = m(input)"
    return argument_parser


def ThresholdConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "Thresholds each element of the input Tensor.\n\nThreshold is defined as:\n\n.. math::\n    y =\n    \\begin{cases}\n    x, &\\text{ if } x > \\text{threshold} \\\\\n    \\text{value}, &\\text{ otherwise }\n    \\end{cases}\n\n\nShape:\n    - Input: :math:`(N, *)` where `*` means, any number of additional\n      dimensions\n    - Output: :math:`(N, *)`, same shape as the input\n\nExamples::\n\n    >>> m = nn.Threshold(0.1, 20)\n    >>> input = torch.randn(2)\n    >>> output = m(input)"
    argument_parser.add_argument(
        "--threshold",
        type=float,
        help="The value to threshold at",
        required=True,
        default=False,
    )
    argument_parser.add_argument(
        "--value", type=float, help="The value to replace with", required=True
    )
    argument_parser.add_argument(
        "--inplace",
        type=bool,
        help="can optionally do the operation in-place.",
        required=True,
        default=False,
    )
    argument_parser.add_argument(
        "--__constants__",
        type=loads,
        help="",
        required=True,
        default="['threshold', 'value', 'inplace']",
    )
    return argument_parser


__all__ = [
    "CELUConfig",
    "ELUConfig",
    "GELUConfig",
    "GLUConfig",
    "HardshrinkConfig",
    "HardsigmoidConfig",
    "HardswishConfig",
    "HardtanhConfig",
    "LeakyReLUConfig",
    "LogSigmoidConfig",
    "LogSoftmaxConfig",
    "MultiheadAttentionConfig",
    "PReLUConfig",
    "RReLUConfig",
    "ReLUConfig",
    "ReLU6Config",
    "SELUConfig",
    "SiLUConfig",
    "SigmoidConfig",
    "SoftmaxConfig",
    "Softmax2dConfig",
    "SoftminConfig",
    "SoftplusConfig",
    "SoftshrinkConfig",
    "SoftsignConfig",
    "TanhConfig",
    "TanhshrinkConfig",
    "ThresholdConfig",
]
