"""
    Implementation of ml_params BaseTrainer API
    """
from functools import partial
from os import path
from typing import Any, Callable, Optional

import torch.cuda
import torch.optim
import torch.utils.data
from ml_params.base import BaseTrainer
from typing_extensions import Literal

from ml_params_pytorch import get_logger
from ml_params_pytorch.ml_params.datasets import (
    load_data_from_torchvision_or_ml_prepare,
)
from ml_params_pytorch.ml_params.type_generators import (
    exposed_losses,
    exposed_optimizer_lr_schedulers,
    exposed_optimizers,
)
from ml_params_pytorch.ml_params.utils import test, train

logger = get_logger(
    ".".join(
        (
            path.basename(path.dirname(__file__)),
            path.basename(__file__).rpartition(".")[0],
        )
    )
)


class TorchTrainer(BaseTrainer):
    """ Implementation of ml_params BaseTrainer for TensorFlow """

    data = None
    get_model: Optional[Callable[[], torch.nn.Module]] = None

    def load_data(
        self,
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
            "ImageFolder",  # Should include this in the README for how to bring in your own data
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
            "VisionDataset",
        ],
        data_loader=load_data_from_torchvision_or_ml_prepare,
        data_type="infer",
        output_type="numpy",
        K=None,
        **data_loader_kwargs
    ):
        """
        Load the data for your ML pipeline. Will be fed into `train`.

        :param dataset_name: name of dataset
        :type dataset_name: ```str```

        :param data_loader: function that returns the expected data type.
         Defaults to TensorFlow Datasets and ml_prepare combined one.
        :type data_loader: ```Callable[[...], Union[Tuple[tf.data.Dataset, tf.data.Dataset],
         Tuple[np.ndarray, np.ndarray], Tuple[Any, Any]]```

        :param data_loader_kwargs: pass this as arguments to data_loader function
        :type data_loader_kwargs: ```**data_loader_kwargs```

        :param data_type: incoming data type
        :type data_type: ```str```

        :param output_type: outgoing data_type, defaults to no conversion
        :type output_type: ```Optional[Literal['numpy']]```

        :param K: backend engine, e.g., `np` or `tf`
        :type K: ```Literal['np', 'tf']```

        :return: Dataset splits (by default, your train and test)
        :rtype: ```Tuple[np.ndarray, np.ndarray]```
        """
        self.data = super(TorchTrainer, self).load_data(
            dataset_name=dataset_name,
            data_loader=data_loader or load_data_from_torchvision_or_ml_prepare,
            data_type=data_type,
            output_type=output_type,
            K=K,
            **data_loader_kwargs
        )
        return self.data

    def load_model(
        self, *, model: Any, call: bool = False, **model_kwargs
    ) -> Callable[[], torch.nn.Module]:
        """
        Load the model.
        Takes a model object, or a pipeline that downloads & configures before returning a model object.

        :param *: syntactic note indicating everything after is a keyword-only argument

        :param model: model object, e.g., a tf.keras.Sequential, tl.Serial,  nn.Module instance

        :param call: whether to call `model()` even if `len(model_kwargs) == 0`

        :param **model_kwargs: to be passed into the model. If empty, doesn't call, unless call=True.

        :return: self.model, e.g., the result of applying `model_kwargs` on model
        """
        self.get_model = (
            lambda: model(**model_kwargs)
            if call is True or len(model_kwargs)
            else model
        )
        return self.get_model

    def train(
        self,
        *,
        epochs: int,
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
        ],
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
        ],
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
        ] = "StepLR",
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
        ] = None,
        metric_emit_freq: Optional[Callable[[int], bool]] = None,
        output_type: str = "infer",
        validation_split: float = 0.1,
        batch_size: int = 128,
        tpu_address: Optional[str] = None,
        **kwargs
    ):
        """
        Run the training loop for your ML pipeline.

        :param epochs: number of epochs (must be greater than 0)

        :param optimizer: Optimizer, can be a string (depending on the framework) or an instance of a class

        :param loss: Loss function, can be a string (depending on the framework) or an instance of a class

        :param lr_scheduler: Learning rate scheduler.
            E.g., `StepLR` will decay the lr of each param group by gamma every step_size epochs.

        :param activation: Activation function

        :param metric_emit_freq: `None` for every epoch. E.g., `eq(mod(epochs, 10), 0)` for every 10. Defaults to None

        :param output_type: `if save_directory is not None` then save in this format, e.g., 'h5'.

        :param validation_split: Optional float between 0 and 1, fraction of data to reserve for validation.

        :param batch_size: batch size at each iteration.

        :param tpu_address: Address of TPU cluster. If None, don't connect & run within TPU context.

        :param kwargs: additional keyword arguments

        :return: the model
        :rtype: ```Any```
        """
        super(TorchTrainer, self).train(epochs=epochs)
        assert self.data is not None
        assert self.get_model is not None
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        data_loader_kwargs = {"batch_size": batch_size}
        if use_cuda:
            data_loader_kwargs.update(
                {"num_workers": 1, "pin_memory": True, "shuffle": True}
            )
        train_loader = torch.utils.data.DataLoader(self.data[0], **data_loader_kwargs)
        test_loader = torch.utils.data.DataLoader(self.data[1], **data_loader_kwargs)
        self.model = self.get_model().to(device)
        optim: torch.optim.Optimizer = (
            partial(acquire_symbols_from(exposed_optimizers, optimizer), lr=1.0)
            if isinstance(optimizer, str)
            else optimizer
        )(self.model.parameters())
        del optimizer
        loss_func: torch.nn.L1Loss = (
            acquire_symbols_from(exposed_losses, loss)()
            if isinstance(loss, str)
            else loss
        )
        del loss
        lr_sched: torch.optim.lr_scheduler.StepLR = (
            partial(
                acquire_symbols_from(exposed_optimizer_lr_schedulers, lr_scheduler),
                gamma=0.7,
                step_size=1,
            )
            if isinstance(lr_scheduler, str)
            else lr_scheduler
        )(optim)
        del lr_scheduler
        common_kwargs = {"device": device, "loss_func": loss_func, "model": self.model}
        print(
            "loss:",
            loss_func,
            ";\noptim:",
            optim,
            ";\nlr_sched:",
            lr_sched,
            ";\nmodel:",
            self.model,
            ";",
        )
        for epoch in range(1, epochs + 1):
            train(
                epoch=epoch,
                metric_emit_freq=metric_emit_freq,
                optimizer=optim,
                train_loader=train_loader,
                **common_kwargs
            )
            test(test_loader=test_loader, **common_kwargs)
            lr_sched.step(None)


def acquire_symbols_from(name2sym, name, never_str=False):
    """
    Acquire the symbol(s) from the iterable

    :param name2sym: Dict from symbol name to symbol
    :type name2sym: ```Dict[Str, Any]```

    :param name: Name of symbol. All namespace is removed.
    :type name: ```Union[Any, str]```

    :param never_str: If True, ensure that `getattr` on the module is always called
    :type never_str: ```bool```

    :return: The list of symbols acquired from the module
    :rtype: ```Optional[List[Union[Any, str]]]```
    """
    if isinstance(name, str):
        name = name.rpartition(".")[2] if name.count(".") > 0 else name
        if name in name2sym:
            return name2sym[name]
    if never_str:
        raise KeyError("{!r} not found in {!r}".format(name, ""))
    return name


del get_logger
__all__ = ["TorchTrainer"]
