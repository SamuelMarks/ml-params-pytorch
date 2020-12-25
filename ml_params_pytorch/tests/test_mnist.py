from os import path
from shutil import rmtree
from tempfile import mkdtemp
from unittest import TestCase
from unittest import main as unittest_main

import torch.nn.functional as F
from torch import optim

from ml_params_pytorch.example_model import Net
from ml_params_pytorch.ml_params_impl import PyTorchTrainer


class TestMnist(TestCase):
    pytorch_datasets_dir = None  # type: str or None
    model_dir = None  # type: str or None

    @classmethod
    def setUpClass(cls) -> None:
        TestMnist.pytorch_datasets_dir = path.join(
            path.expanduser("~"), "pytorch_datasets"
        )
        TestMnist.model_dir = mkdtemp("_model_dir")

    @classmethod
    def tearDownClass(cls) -> None:
        # rmtree(TestMnist.pytorch_datasets_dir)
        rmtree(TestMnist.model_dir)

    def test_mnist(self) -> None:
        num_classes = 10
        epochs = 3

        trainer = PyTorchTrainer()
        trainer.load_data(
            "MNIST",
            datasets_dir=TestMnist.pytorch_datasets_dir,
            num_classes=num_classes,
        )
        trainer.load_model(Net, call=True)
        trainer.train(
            epochs=epochs,
            model_dir=TestMnist.model_dir,
            optimizer=optim.Adadelta,
            loss=F.nll_loss,
            metrics=None,
            callbacks=None,
            save_directory=None,
            metric_emit_freq=lambda batch_idx: batch_idx % 10 == 0,
        )


if __name__ == "__main__":
    unittest_main()
