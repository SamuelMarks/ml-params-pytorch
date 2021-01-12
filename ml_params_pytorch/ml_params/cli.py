"""
CLI interface to ml-params-pytorch. Expected to be bootstrapped by ml-params.
"""

from yaml import safe_load as loads


def train_parser(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser, the model
    :rtype: ```Tuple[ArgumentParser, Any]```
    """
    argument_parser.description = "Run the training loop for your ML pipeline."
    argument_parser.add_argument(
        "--epochs", help="number of epochs (must be greater than 0)", required=True
    )
    argument_parser.add_argument(
        "--optimizer",
        help="Optimizer, can be a string (depending on the framework) or an instance of a class",
        required=True,
    )
    argument_parser.add_argument(
        "--loss",
        help="Loss function, can be a string (depending on the framework) or an instance of a class",
        required=True,
    )
    argument_parser.add_argument(
        "--lr_scheduler",
        type=str,
        help="""Learning rate scheduler.
    E.g., `StepLR` will decay the lr of each param group by gamma every step_size epochs.""",
        required=True,
        default="StepLR",
    )
    argument_parser.add_argument("--activation", type=str, help="Activation function")
    argument_parser.add_argument(
        "--metric_emit_freq",
        type=int,
        help="`None` for every epoch. E.g., `eq(mod(epochs, 10), 0)` for every 10.",
    )
    argument_parser.add_argument(
        "--output_type",
        type=str,
        help="`if save_directory is not None` then save in this format, e.g., 'h5'.",
        required=True,
        default="infer",
    )
    argument_parser.add_argument(
        "--validation_split",
        type=float,
        help="Optional float between 0 and 1, fraction of data to reserve for validation.",
        default=0.1,
    )
    argument_parser.add_argument(
        "--batch_size",
        type=int,
        help="batch size at each iteration.",
        required=True,
        default=128,
    )
    argument_parser.add_argument(
        "--tpu_address",
        type=str,
        help="Address of TPU cluster. If None, don't connect & run within TPU context.",
    )
    argument_parser.add_argument(
        "--kwargs", type=loads, help="additional keyword arguments"
    )
    return argument_parser


def load_data_parser(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser, Dataset splits (by default, your train and test)
    :rtype: ```Tuple[ArgumentParser, Tuple[np.ndarray, np.ndarray]]```
    """
    argument_parser.description = (
        "Load the data for your ML pipeline. Will be fed into `train`."
    )
    argument_parser.add_argument(
        "--dataset_name",
        type=str,
        help="name of dataset",
        required=True,
        default="load_data_from_torchvision_or_ml_prepare",
    )
    argument_parser.add_argument(
        "--data_loader",
        type=str,
        help="""function that returns the expected data type.
""",
        required=True,
        default="infer",
    )
    argument_parser.add_argument(
        "--data_type",
        type=str,
        help="incoming data type",
        required=True,
        default="numpy",
    )
    argument_parser.add_argument("--output_type", type=str, help="outgoing data_type,")
    argument_parser.add_argument(
        "--K",
        choices=("np", "tf"),
        help="backend engine, e.g., `np` or `tf`",
        required=True,
    )
    argument_parser.add_argument(
        "--data_loader_kwargs",
        type=loads,
        help="pass this as arguments to data_loader function",
    )
    return argument_parser, "```self.data```"


def load_model_parser(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser, self.model, e.g., the result of applying `model_kwargs` on model
    :rtype: ```Tuple[ArgumentParser, Callable[[], torch.nn.Module]]```
    """
    argument_parser.description = """Load the model.
Takes a model object, or a pipeline that downloads & configures before returning a model object."""
    argument_parser.add_argument(
        "--model",
        help="model object, e.g., a tf.keras.Sequential, tl.Serial,  nn.Module instance",
        required=True,
    )
    argument_parser.add_argument(
        "--call",
        type=bool,
        help="whether to call `model()` even if `len(model_kwargs) == 0`",
        required=True,
        default=False,
    )
    argument_parser.add_argument(
        "--model_kwargs",
        type=loads,
        help="to be passed into the model. If empty, doesn't call, unless call=True.",
    )
    return argument_parser, "```self.get_model```"


__all__ = ["load_data_parser", "load_model_parser", "train_parser"]
