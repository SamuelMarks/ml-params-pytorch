from typing import Any, Callable, Literal, Optional, Tuple, Union

try:
    import tensorflow as tf
except ImportError:
    tf = None
try:
    import numpy as np
except ImportError:
    np = None
import torch.nn


class ConfigClass(object):
    """
    Run the training loop for your ML pipeline.

    :cvar epochs: number of epochs (must be greater than 0)
    :cvar optimizer: Optimizer, can be a string (depending on the framework) or an instance of a class
    :cvar loss: Loss function, can be a string (depending on the framework) or an instance of a class
    :cvar lr_scheduler: Learning rate scheduler.
    E.g., `StepLR` will decay the lr of each param group by gamma every step_size epochs.
    :cvar activation: Activation function
    :cvar metric_emit_freq: `None` for every epoch. E.g., `eq(mod(epochs, 10), 0)` for every 10.
    :cvar output_type: `if save_directory is not None` then save in this format, e.g., 'h5'.
    :cvar validation_split: Optional float between 0 and 1, fraction of data to reserve for validation.
    :cvar batch_size: batch size at each iteration.
    :cvar tpu_address: Address of TPU cluster. If None, don't connect & run within TPU context.
    :cvar kwargs: additional keyword arguments
    :cvar return_type: the model"""

    epochs: int = 0
    optimizer: Any = None
    loss: Any = None
    lr_scheduler: str = "StepLR"
    activation: Any = None
    metric_emit_freq: Optional[Callable[[int], bool]] = None
    output_type: str = "infer"
    validation_split: Optional[float] = 0.1
    batch_size: int = 128
    tpu_address: Optional[str] = None
    kwargs: Optional[dict] = None
    return_type: Any = None


class LoadDataConfig(object):
    """
    Load the data for your ML pipeline. Will be fed into `train`.

    :cvar dataset_name: name of dataset
    :cvar data_loader: function that returns the expected data type.

    :cvar data_type: incoming data type
    :cvar output_type: outgoing data_type,
    :cvar K: backend engine, e.g., `np` or `tf`
    :cvar data_loader_kwargs: pass this as arguments to data_loader function
    :cvar return_type: Dataset splits (by default, your train and test)"""

    dataset_name: Literal[
        "CIFAR10",
        "CIFAR100",
        "Caltech101",
        "Caltech256",
        "CelebA",
        "Cityscapes",
        "CocoCaptions",
        "CocoDetection",
        "DatasetFolder",
        "EMNIST",
        "FakeData",
        "FashionMNIST",
        "Flickr30k",
        "Flickr8k",
        "HMDB51",
        "ImageFolder",
        "ImageNet",
        "KMNIST",
        "Kinetics400",
        "LSUN",
        "LSUNClass",
        "MNIST",
        "Omniglot",
        "PhotoTour",
        "Places365",
        "QMNIST",
        "SBDataset",
        "SBU",
        "SEMEION",
        "STL10",
        "SVHN",
        "UCF101",
        "USPS",
        "VOCDetection",
        "VOCSegmentation",
    ] = "load_data_from_torchvision_or_ml_prepare"
    data_loader: Callable[
        [...],
        Union[
            Tuple[tf.data.Dataset, tf.data.Dataset],
            Tuple[np.ndarray, np.ndarray],
            Tuple[Any, Any],
        ],
    ] = "infer"
    data_type: str = "numpy"
    output_type: Optional[Literal["numpy"]] = None
    K: Literal["np", "tf"] = None
    data_loader_kwargs: Optional[dict] = None
    return_type: Tuple[np.ndarray, np.ndarray] = "``````self.data``````"


class LoadModelConfig(object):
    """
    Load the model.
    Takes a model object, or a pipeline that downloads & configures before returning a model object.

    :cvar model: model object, e.g., a tf.keras.Sequential, tl.Serial,  nn.Module instance
    :cvar call: whether to call `model()` even if `len(model_kwargs) == 0`
    :cvar model_kwargs: to be passed into the model. If empty, doesn't call, unless call=True.
    :cvar return_type: self.model, e.g., the result of applying `model_kwargs` on model"""

    model: Any = None
    call: bool = False
    model_kwargs: Optional[dict] = None
    return_type: Callable[[], torch.nn.Module] = "``````self.get_model``````"


__all__ = ["ConfigClass", "LoadDataConfig", "LoadModelConfig"]


class TrainConfig(object):
    """
    Run the training loop for your ML pipeline.

    :cvar epochs: number of epochs (must be greater than 0)
    :cvar optimizer: Optimizer, can be a string (depending on the framework) or an instance of a class
    :cvar loss: Loss function, can be a string (depending on the framework) or an instance of a class
    :cvar lr_scheduler: Learning rate scheduler.
    E.g., `StepLR` will decay the lr of each param group by gamma every step_size epochs.
    :cvar activation: Activation function
    :cvar metric_emit_freq: `None` for every epoch. E.g., `eq(mod(epochs, 10), 0)` for every 10.
    :cvar output_type: `if save_directory is not None` then save in this format, e.g., 'h5'.
    :cvar validation_split: Optional float between 0 and 1, fraction of data to reserve for validation.
    :cvar batch_size: batch size at each iteration.
    :cvar tpu_address: Address of TPU cluster. If None, don't connect & run within TPU context.
    :cvar kwargs: additional keyword arguments
    :cvar return_type: the model"""

    epochs: int = 0
    optimizer: Literal[
        "ASGD",
        "Adadelta",
        "Adagrad",
        "Adam",
        "AdamW",
        "Adamax",
        "LBFGS",
        "RMSprop",
        "Rprop",
        "SGD",
        "SparseAdam",
    ] = None
    loss: Literal[
        "CosineAnnealingLR",
        "CosineAnnealingWarmRestarts",
        "Counter",
        "CyclicLR",
        "ExponentialLR",
        "LambdaLR",
        "MultiStepLR",
        "MultiplicativeLR",
        "OneCycleLR",
        "ReduceLROnPlateau",
        "StepLR",
    ] = None
    lr_scheduler: Literal[
        "BCELoss",
        "BCEWithLogitsLoss",
        "CTCLoss",
        "CosineEmbeddingLoss",
        "CrossEntropyLoss",
        "HingeEmbeddingLoss",
        "KLDivLoss",
        "L1Loss",
        "MSELoss",
        "MarginRankingLoss",
        "MultiLabelMarginLoss",
        "MultiLabelSoftMarginLoss",
        "MultiMarginLoss",
        "NLLLoss",
        "NLLLoss2d",
        "PairwiseDistance",
        "PoissonNLLLoss",
        "SmoothL1Loss",
        "SoftMarginLoss",
        "TripletMarginLoss",
        "TripletMarginWithDistanceLoss",
    ] = "StepLR"
    activation: Literal[
        "CELU",
        "ELU",
        "GELU",
        "GLU",
        "Hardshrink",
        "Hardsigmoid",
        "Hardswish",
        "Hardtanh",
        "LeakyReLU",
        "LogSigmoid",
        "LogSoftmax",
        "MultiheadAttention",
        "PReLU",
        "RReLU",
        "ReLU",
        "ReLU6",
        "SELU",
        "SiLU",
        "Sigmoid",
        "Softmax",
        "Softmax2d",
        "Softmin",
        "Softplus",
        "Softshrink",
        "Softsign",
        "Tanh",
        "Tanhshrink",
        "Threshold",
    ] = None
    metric_emit_freq: Optional[Callable[[int], bool]] = None
    output_type: str = "infer"
    validation_split: Optional[float] = 0.1
    batch_size: int = 128
    tpu_address: Optional[str] = None
    kwargs: Optional[dict] = None
    return_type: Any = None
