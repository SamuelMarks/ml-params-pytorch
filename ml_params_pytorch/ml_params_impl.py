""" Implementation of ml_params API """

# Mostly based off https://github.com/keras-team/keras-io/blob/8320a6c/examples/vision/mnist_convnet.py

from os import path, environ
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from ml_params.base import BaseTrainer
from ml_prepare.datasets import datasets2classes
from ml_prepare.exectors import build_tfds_dataset
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from ml_params_pytorch import get_logger
from ml_params_pytorch.utils import train, test

if environ.get('TF_KERAS', True):
    pass
else:
    pass

logger = get_logger('.'.join((path.basename(path.dirname(__file__)),
                              path.basename(__file__).rpartition('.')[0])))


class TensorFlowTrainer(BaseTrainer):
    """ Implementation of ml_params BaseTrainer for TensorFlow """

    data = None  # type: (None or Tuple[tf.data.Dataset, tf.data.Dataset] )
    model = None  # contains the model, e.g., a `tl.Serial`

    def __init__(self, model, **model_kwargs):
        super(TensorFlowTrainer, self).__init__()
        self.model = model(**model_kwargs)

    def load_data(self, dataset_name, data_loader=None,
                  data_loader_kwargs=None, data_type='infer',
                  output_type=None, K=None):
        """
        Load the data for your ML pipeline. Will be fed into `train`.

        :param dataset_name: name of dataset
        :type dataset_name: ```str```

        :param data_loader: function that returns the expected data type.
         Defaults to TensorFlow Datasets and ml_prepare combined one.
        :type data_loader: ```None or (*args, **kwargs) -> tf.data.Datasets or Any```

        :param data_loader_kwargs: pass this as arguments to data_loader function
        :type data_loader_kwargs: ```None or dict```

        :param data_type: incoming data type, defaults to 'infer'
        :type data_type: ```str```

        :param output_type: outgoing data_type, defaults to no conversion
        :type output_type: ```None or 'numpy'```

        :param K: backend engine, e.g., `np` or `tf`
        :type K: ```None or np or tf or Any```

        :return: Dataset splits (by default, your train and test)
        :rtype: ```Tuple[tf.data.Dataset, tf.data.Dataset] or Tuple[np.ndarray, np.ndarray]```
        """
        self.data = super(TensorFlowTrainer, self).load_data(
            dataset_name=dataset_name,
            data_loader=data_loader or self.load_data_from_torchvision_or_ml_prepare,
            data_loader_kwargs=data_loader_kwargs,
            data_type=data_type,
            output_type=output_type
        )

    @staticmethod
    def load_data_from_torchvision_or_ml_prepare(dataset_name, pytorch_datasets_dir=None, data_loader_kwargs=None):
        """
        Acquire from the official keras model zoo, or the ophthalmology focussed ml-prepare library

        :param dataset_name: name of dataset
        :type dataset_name: ```str```

        :param pytorch_datasets_dir: directory to look for models in. Default is ~/pytorch_datasets.
        :type pytorch_datasets_dir: ```None or str```

        :param data_loader_kwargs: pass this as arguments to data_loader function
        :type data_loader_kwargs: ```None or dict```

        :return: Train and tests dataset splits
        :rtype: ```Tuple[tf.data.Dataset, tf.data.Dataset] or Tuple[np.ndarray, np.ndarray]```
        """
        data_loader_kwargs.update({
            'dataset_name': dataset_name,
            'tfds_dir': pytorch_datasets_dir,

        })
        if 'scale' not in data_loader_kwargs:
            data_loader_kwargs['scale'] = 255

        if dataset_name in datasets2classes:
            ds_builder = build_tfds_dataset(**data_loader_kwargs)

            if hasattr(ds_builder, 'download_and_prepare_kwargs'):
                download_and_prepare_kwargs = getattr(ds_builder, 'download_and_prepare_kwargs')
                delattr(ds_builder, 'download_and_prepare_kwargs')
            else:
                download_and_prepare_kwargs = None

            return BaseTrainer.common_dataset_handler(
                ds_builder=ds_builder,
                download_and_prepare_kwargs=download_and_prepare_kwargs,
                scale=None, K=None, as_numpy=True
            )
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            return (datasets.MNIST(pytorch_datasets_dir, train=True, download=True,
                                   transform=transform),
                    datasets.MNIST(pytorch_datasets_dir, train=False,
                                   transform=transform))

    def train(self, epochs, model=None, batch_size=64, test_batch_size=1000, lr=1.0,
              gamma=0.7, no_cuda=True, seed=1, log_interval=10, save_model=False,
              *args, **kwargs):
        super(TensorFlowTrainer, self).train(epochs=epochs, *args, **kwargs)
        assert self.data is not None
        assert self.model is not None
        if model is not None:
            self.model = model

        use_cuda = torch.cuda.is_available()

        device = torch.device("cuda" if use_cuda else "cpu")

        kwargs = {'batch_size': batch_size}
        if use_cuda:
            kwargs.update({'num_workers': 1,
                           'pin_memory': True,
                           'shuffle': True},
                          )

        train_loader = torch.utils.data.DataLoader(self.data[0], **kwargs)
        test_loader = torch.utils.data.DataLoader(self.data[1], **kwargs)

        self.model = self.model.to(device)
        optimizer = optim.Adadelta(self.model.parameters(), lr=lr)

        scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
        for epoch in range(1, epochs + 1):
            train(self.model, device, train_loader, optimizer, epoch, log_interval)
            test(self.model, device, test_loader)
            scheduler.step(None)


del Tuple, build_tfds_dataset, get_logger

__all__ = ['TensorFlowTrainer']
