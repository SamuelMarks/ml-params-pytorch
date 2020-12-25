from importlib import import_module
from pkgutil import find_loader

from ml_params.datasets import load_data_from_ml_prepare
from torchvision import datasets, transforms

datasets2classes = (
    {}
    if find_loader("ml_prepare") is None or find_loader("tensorflow_datasets") is None
    else getattr(import_module("ml_prepare.datasets"), "datasets2classes")
)

def load_data_from_torchvision_or_ml_prepare(
    dataset_name, datasets_dir=None, K=None, as_numpy=True, **data_loader_kwargs
):
    """
    Acquire from the official torchvision model zoo, or the ophthalmology focussed ml-prepare library

    :param dataset_name: name of dataset
    :type dataset_name: ```str```

    :param datasets_dir: directory to look for models in. Default is ~/tensorflow_datasets.
    :type datasets_dir: ```None or str```

    :param K: backend engine, e.g., `np` or `tf`
    :type K: ```Literal['np', 'tf']```

    :param as_numpy: Convert to numpy ndarrays
    :type as_numpy: ```bool```

    :param data_loader_kwargs: pass this as arguments to data_loader function
    :type data_loader_kwargs: ```**data_loader_kwargs```

    :return: Train and tests dataset splits
    :rtype: ```Tuple[np.ndarray, np.ndarray]```
    """
    if dataset_name in datasets2classes:
        return load_data_from_ml_prepare(
            dataset_name=dataset_name,
            tfds_dir=datasets_dir,
            as_numpy=as_numpy,
            **data_loader_kwargs
        )

    data_loader_kwargs.update(
        {
            "dataset_name": dataset_name,
            "datasets_dir": datasets_dir,
        }
    )
    if "scale" not in data_loader_kwargs:
        data_loader_kwargs["scale"] = 255
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset = getattr(datasets, dataset_name)
    return (
        dataset(datasets_dir, train=True, download=True, transform=transform),
        dataset(datasets_dir, train=False, transform=transform),
    )
