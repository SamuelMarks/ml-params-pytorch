from os import path
from shutil import rmtree
from tempfile import mkdtemp
from typing import Optional
from unittest import TestCase
from unittest import main as unittest_main

import torch.nn.functional as F

from ml_params_pytorch.ml_params.trainer import TorchTrainer
from ml_params_pytorch.tests.example_model import Net


class TestMnist(TestCase):
    pytorch_datasets_dir: Optional[str] = None
    model_dir: Optional[str] = None

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
        num_classes: int = 10
        epochs: int = 3

        trainer: TorchTrainer = TorchTrainer()
        trainer.load_data(
            "MNIST",
            tfds_dir=TestMnist.pytorch_datasets_dir,
            # num_classes=num_classes,
        )
        trainer.load_model(model=Net, call=True)
        trainer.train(
            epochs=epochs,
            model_dir=TestMnist.model_dir,
            optimizer="Adadelta",
            loss=F.nll_loss,
            save_directory="/tmp",
            metric_emit_freq=lambda batch_idx: batch_idx % 10 == 0,
        )


if __name__ == "__main__":
    unittest_main()
