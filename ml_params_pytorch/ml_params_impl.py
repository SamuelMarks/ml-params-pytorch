""" Implementation of ml_params API """

# Mostly based off https://github.com/keras-team/keras-io/blob/8320a6c/examples/vision/mnist_convnet.py

from os import path
from sys import stdout
from typing import Tuple

import numpy as np
import torch
from ml_params.base import BaseTrainer
from torch.optim.lr_scheduler import StepLR

from ml_params_pytorch import get_logger
from ml_params_pytorch.datasets import load_data_from_torchvision_or_ml_prepare
from ml_params_pytorch.utils import train, test

logger = get_logger('.'.join((path.basename(path.dirname(__file__)),
                              path.basename(__file__).rpartition('.')[0])))


class PyTorchTrainer(BaseTrainer):
    """ Implementation of ml_params BaseTrainer for PyTorchTrainer """

    data = None  # type: (None or Tuple[np.ndarry, np.ndarray] )
    model = None  # contains the model, e.g., a `tl.Serial`

    def load_data(self, dataset_name, data_loader=load_data_from_torchvision_or_ml_prepare,
                  data_type='infer', output_type='numpy', K=None,
                  **data_loader_kwargs):
        """
        Load the data for your ML pipeline. Will be fed into `train`.

        :param dataset_name: name of dataset
        :type dataset_name: ```str```

        :param data_loader: function that returns the expected data type.
         Defaults to TensorFlow Datasets and ml_prepare combined one.
        :type data_loader: ```None or (*args, **kwargs) -> tf.data.Datasets or Any```

        :param data_loader_kwargs: pass this as arguments to data_loader function
        :type data_loader_kwargs: ```**data_loader_kwargs```

        :param data_type: incoming data type, defaults to 'infer'
        :type data_type: ```str```

        :param output_type: outgoing data_type, defaults to no conversion
        :type output_type: ```None or 'numpy'```

        :param K: backend engine, e.g., `np` or `tf`
        :type K: ```None or np or tf or Any```

        :return: Dataset splits (by default, your train and test)
        :rtype: ```Tuple[np.ndarray, np.ndarray]```
        """
        self.data = super(PyTorchTrainer, self).load_data(dataset_name=dataset_name,
                                                          data_loader=data_loader,
                                                          data_type=data_type,
                                                          output_type=output_type,
                                                          K=K,
                                                          **data_loader_kwargs)

    def train(self, callbacks, epochs, loss, metrics, metric_emit_freq, optimizer,
              save_directory, output_type='infer', writer=stdout,
              batch_size=64, test_batch_size=1000, lr=1.0,
              gamma=0.7, no_cuda=True, seed=1,
              *args, **kwargs):
        """
        Run the training loop for your ML pipeline.

        :param callbacks: Collection of callables that are run inside the training loop
        :type callbacks: ```None or List[Callable] or Tuple[Callable]```

        :param epochs: number of epochs (must be greater than 0)
        :type epochs: ```int```

        :param loss: Loss function, can be a string (depending on the framework) or an instance of a class
        :type loss: ```str or Callable or Any```

        :param metrics: Collection of metrics to monitor, e.g., accuracy, f1
        :type metrics: ```None or List[Callable or str] or Tuple[Callable or str]```

        :param metric_emit_freq: Frequency of metric emission, e.g., `lambda: epochs % 10 == 0`, defaults to every epoch
        :type metric_emit_freq: ```None or (*args, **kwargs) -> bool```

        :param optimizer: Optimizer, can be a string (depending on the framework) or an instance of a class
        :type callbacks: ```str or Callable or Any```

        :param save_directory: Directory to save output in, e.g., weights in h5 files. If None, don't save.
        :type save_directory: ```None or str```

        :param output_type: `if save_directory is not None` then save in this format, e.g., 'h5'.
        :type output_type: ```str```

        :param writer: Writer for all output, could be a TensorBoard instance, a file handler like stdout or stderr
        :type writer: ```stdout or Any```

        :param batch_size:
        :type batch_size: ```int```

        :param test_batch_size:
        :type test_batch_size: ```int```

        :param lr: learning rate
        :type lr: ```int```

        :param gamma:
        :type gamma: ```float```

        :param no_cuda:
        :type no_cuda: ```bool```

        :param seed:
        :type seed: ```int```

        :param args:
        :param kwargs:
        :return:
        """
        super(PyTorchTrainer, self).train(callbacks=callbacks,
                                          epochs=epochs,
                                          loss=loss,
                                          metrics=metrics,
                                          metric_emit_freq=metric_emit_freq,
                                          optimizer=optimizer,
                                          save_directory=save_directory,
                                          output_type=output_type,
                                          writer=writer,
                                          *args, **kwargs)
        assert self.data is not None
        assert self.model is not None

        use_cuda = torch.cuda.is_available()

        device = torch.device("cuda" if use_cuda else "cpu")

        data_loader_kwargs = {'batch_size': batch_size}
        if use_cuda:
            data_loader_kwargs.update({
                'num_workers': 1,
                'pin_memory': True,
                'shuffle': True
            })

        train_loader = torch.utils.data.DataLoader(self.data[0], **data_loader_kwargs)
        test_loader = torch.utils.data.DataLoader(self.data[1], **data_loader_kwargs)

        self.model = self.model.to(device)
        optimizer = optimizer(self.model.parameters(), lr=lr)

        scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
        common_kwargs = {'model': self.model, 'device': device, 'loss_func': loss}
        for epoch in range(1, epochs + 1):
            train(train_loader=train_loader, epoch=epoch, metric_emit_freq=metric_emit_freq,
                  optimizer=optimizer, **common_kwargs)
            test(test_loader=test_loader, **common_kwargs)
            scheduler.step(None)


del Tuple, get_logger

__all__ = ['PyTorchTrainer']
