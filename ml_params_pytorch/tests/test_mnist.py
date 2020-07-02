from os import path
from shutil import rmtree
from tempfile import mkdtemp
from unittest import TestCase, main as unittest_main

from ml_params_pytorch.example_model import Net
from ml_params_pytorch.ml_params_impl import TensorFlowTrainer


class TestMnist(TestCase):
    pytorch_datasets_dir = None  # type: str or None
    model_dir = None  # type: str or None

    @classmethod
    def setUpClass(cls) -> None:
        TestMnist.pytorch_datasets_dir = path.join(path.expanduser('~'), 'pytorch_datasets')
        TestMnist.model_dir = mkdtemp('_model_dir')

    @classmethod
    def tearDownClass(cls) -> None:
        # rmtree(TestMnist.pytorch_datasets_dir)
        rmtree(TestMnist.model_dir)

    def test_mnist(self) -> None:
        num_classes = 10
        trainer = TensorFlowTrainer(Net)
        trainer.load_data('mnist', data_loader_kwargs={
            'pytorch_datasets_dir': TestMnist.pytorch_datasets_dir,
            'data_loader_kwargs': {'num_classes': num_classes}
        })

        epochs = 3
        trainer.train(epochs=epochs, model_dir=TestMnist.model_dir)


if __name__ == '__main__':
    unittest_main()
