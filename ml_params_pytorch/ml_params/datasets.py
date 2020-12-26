""" Generated Dataset CLI parsers """
import pickle
from importlib import import_module
from pkgutil import find_loader

from ml_params.datasets import load_data_from_ml_prepare
from torchvision import datasets, transforms
from yaml import safe_load as loads

datasets2classes = (
    {}
    if any(
        map(
            lambda mod: find_loader(mod) is None,
            ("ml_prepare", "tensorflow_datasets", "tensorflow"),
        )
    )
    else getattr(import_module("ml_prepare.datasets"), "datasets2classes")
)


def load_data_from_torchvision_or_ml_prepare(
    dataset_name, datasets_dir=None, K=None, as_numpy=True, **data_loader_kwargs
):
    """
    Acquire from the official torchvision model zoo, or the ophthalmology focussed ml-prepare library

    :param dataset_name: name of dataset
    :type dataset_name: ```str```

    :param datasets_dir: directory to look for models in. Default is ~/pytorch_datasets.
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


def CIFAR10Config(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = (
        "`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset."
    )
    argument_parser.add_argument(
        "--root",
        type=bool,
        help="Root directory of dataset where directory\n        ``cifar-10-batches-py`` exists or will be saved to if download is set to True.",
        required=True,
        default=True,
    )
    argument_parser.add_argument(
        "--train",
        type=bool,
        help="If True, creates dataset from training set, otherwise\n        creates from test set.",
        required=True,
        default=True,
    )
    argument_parser.add_argument(
        "--transform",
        type=str,
        help="A function/transform that takes in an PIL image\n        and returns a transformed version. E.g, ``transforms.RandomCrop``",
    )
    argument_parser.add_argument(
        "--target_transform",
        type=bool,
        help="A function/transform that takes in the\n        target and transforms it.",
        default=False,
    )
    argument_parser.add_argument(
        "--download",
        type=bool,
        help="If true, downloads the dataset from the internet and\n        puts it in root directory. If dataset is already downloaded, it is not\n        downloaded again.",
        required=True,
        default=False,
    )
    argument_parser.add_argument(
        "--filename", type=str, help="", required=True, default="cifar-10-python.tar.gz"
    )
    argument_parser.add_argument(
        "--test_list",
        type=loads,
        action="append",
        help="",
        required=True,
        default="['test_batch', '40351d587109b95175f43aff81a1287e']",
    )
    argument_parser.add_argument(
        "--tgz_md5",
        type=str,
        help="",
        required=True,
        default="c58f30108f718f92721af3b95e74349a",
    )
    argument_parser.add_argument(
        "--meta",
        type=loads,
        help="",
        required=True,
        default="{'filename': 'batches.meta', 'key': 'label_names', 'md5': '5ff9c542aee3614f3951f8cda6e48888'}",
    )
    argument_parser.add_argument(
        "--url",
        type=str,
        help="",
        required=True,
        default="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    )
    argument_parser.add_argument(
        "--train_list",
        type=loads,
        help="",
        required=True,
        default="[['data_batch_1', 'c99cafc152244af753f735de768cd75f'], ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'], ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'], ['data_batch_4', '634d18415352ddfa80567beed471001a'], ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb']]",
    )
    argument_parser.add_argument(
        "--base_folder", type=str, help="", required=True, default="cifar-10-batches-py"
    )
    return argument_parser


def CIFAR100Config(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.\n\nThis is a subclass of the `CIFAR10` Dataset."
    argument_parser.add_argument(
        "--base_folder", type=str, help="", required=True, default="cifar-100-python"
    )
    argument_parser.add_argument(
        "--url",
        type=str,
        help="",
        required=True,
        default="https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
    )
    argument_parser.add_argument(
        "--filename",
        type=str,
        help="",
        required=True,
        default="cifar-100-python.tar.gz",
    )
    argument_parser.add_argument(
        "--tgz_md5",
        type=str,
        help="",
        required=True,
        default="eb9058c3a382ffc7106e4002c42a8d85",
    )
    argument_parser.add_argument(
        "--train_list",
        type=loads,
        action="append",
        help="",
        required=True,
        default="['train', '16019d7e3df5f24257cddd939b257f8d']",
    )
    argument_parser.add_argument(
        "--test_list",
        type=loads,
        action="append",
        help="",
        required=True,
        default="['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc']",
    )
    argument_parser.add_argument(
        "--meta",
        type=loads,
        help="",
        required=True,
        default="{'filename': 'meta', 'key': 'fine_label_names', 'md5': '7973b15100ade9c7d40fb424638fde48'}",
    )
    return argument_parser


def Caltech101Config(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "`Caltech 101 <http://www.vision.caltech.edu/Image_Datasets/Caltech101/>`_ Dataset.\n\n.. warning::\n\n    This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format."
    argument_parser.add_argument(
        "--root",
        type=str,
        help="Root directory of dataset where directory\n        ``caltech101`` exists or will be saved to if download is set to True.",
        required=True,
        default="category",
    )
    argument_parser.add_argument(
        "--target_type",
        type=str,
        help="Type of target to use, ``category`` or",
        required=True,
        default="category",
    )
    argument_parser.add_argument("--transform", type=str, help=None)
    argument_parser.add_argument("--download", type=bool, help=None, required=True)
    argument_parser.add_argument(
        "--target_transform", type=bool, help=None, default=False
    )
    return argument_parser


def Caltech256Config(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "`Caltech 256 <http://www.vision.caltech.edu/Image_Datasets/Caltech256/>`_ Dataset."
    argument_parser.add_argument(
        "--root",
        type=str,
        help="Root directory of dataset where directory\n        ``caltech256`` exists or will be saved to if download is set to True.",
        required=True,
    )
    argument_parser.add_argument(
        "--transform",
        type=str,
        help="A function/transform that takes in an PIL image\n        and returns a transformed version. E.g, ``transforms.RandomCrop``",
    )
    argument_parser.add_argument(
        "--target_transform",
        type=bool,
        help="A function/transform that takes in the\n        target and transforms it.",
        default=False,
    )
    argument_parser.add_argument(
        "--download",
        type=bool,
        help="If true, downloads the dataset from the internet and\n        puts it in root directory. If dataset is already downloaded, it is not\n        downloaded again.",
        required=True,
        default=False,
    )
    return argument_parser


def CelebAConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "`Large-scale CelebFaces Attributes (CelebA) Dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ Dataset."
    argument_parser.add_argument(
        "--root",
        type=str,
        help="Root directory where images are downloaded to.",
        required=True,
        default="train",
    )
    argument_parser.add_argument(
        "--split",
        type=str,
        help="One of {'train', 'valid', 'test', 'all'}.\n        Accordingly dataset is selected.",
        required=True,
        default="train",
    )
    argument_parser.add_argument(
        "--target_type",
        type=str,
        help="Type of target to use, ``attr``, ``identity``, ``bbox``,\n        or ``landmarks``. Can also be a list to output a tuple with all specified target types.\n        The targets represent:\n            ``attr`` (np.array shape=(40,) dtype=int): binary (0, 1) labels for attributes\n            ``identity`` (int): label for each person (data points with the same identity are the same person)\n            ``bbox`` (np.array shape=(4,) dtype=int): bounding box (x, y, width, height)\n            ``landmarks`` (np.array shape=(10,) dtype=int): landmark points (lefteye_x, lefteye_y, righteye_x,\n                righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y)\n       If empty, ``None`` will be returned as target.",
        required=True,
        default="attr",
    )
    argument_parser.add_argument(
        "--transform",
        type=str,
        help="A function/transform that  takes in an PIL image\n        and returns a transformed version. E.g, ``transforms.ToTensor``",
    )
    argument_parser.add_argument(
        "--target_transform",
        type=bool,
        help="A function/transform that takes in the\n        target and transforms it.",
        default=False,
    )
    argument_parser.add_argument(
        "--download",
        type=bool,
        help="If true, downloads the dataset from the internet and\n        puts it in root directory. If dataset is already downloaded, it is not\n        downloaded again.",
        required=True,
        default=False,
    )
    argument_parser.add_argument(
        "--file_list",
        type=loads,
        help="",
        required=True,
        default="[('0B7EVK8r0v71pZjFTYXZWM3FlRnM', '00d2c5bc6d35e252742224ab0c1e8fcb', 'img_align_celeba.zip'), ('0B7EVK8r0v71pblRyaVFSWGxPY0U', '75e246fa4810816ffd6ee81facbd244c', 'list_attr_celeba.txt'), ('1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS', '32bd1bd63d3c78cd57e08160ec5ed1e2', 'identity_CelebA.txt'), ('0B7EVK8r0v71pbThiMVRxWXZ4dU0', '00566efa6fedff7a56946cd1c10f1c16', 'list_bbox_celeba.txt'), ('0B7EVK8r0v71pd0FJY3Blby1HUTQ', 'cc24ecafdb5b50baae59b03474781f8c', 'list_landmarks_align_celeba.txt'), ('0B7EVK8r0v71pY0NSMzRuSXJEVkk', 'd32c9cbf5e040fd4025c592c306e6668', 'list_eval_partition.txt')]",
    )
    argument_parser.add_argument(
        "--base_folder", type=str, help="", required=True, default="celeba"
    )
    return argument_parser


def CityscapesConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "`Cityscapes <http://www.cityscapes-dataset.com/>`_ Dataset.\n\n\nExamples:\n\n    Get semantic segmentation target\n\n    .. code-block:: python\n\n        dataset = Cityscapes('./data/cityscapes', split='train', mode='fine',\n                             target_type='semantic')\n\n        img, smnt = dataset[0]\n\n    Get multiple targets\n\n    .. code-block:: python\n\n        dataset = Cityscapes('./data/cityscapes', split='train', mode='fine',\n                             target_type=['instance', 'color', 'polygon'])\n\n        img, (inst, col, poly) = dataset[0]\n\n    Validate on the \"coarse\" set\n\n    .. code-block:: python\n\n        dataset = Cityscapes('./data/cityscapes', split='val', mode='coarse',\n                             target_type='semantic')\n\n        img, smnt = dataset[0]"
    argument_parser.add_argument(
        "--root",
        type=str,
        help="Root directory of dataset where directory ``leftImg8bit``\n        and ``gtFine`` or ``gtCoarse`` are located.",
        required=True,
        default="train",
    )
    argument_parser.add_argument(
        "--split",
        type=str,
        help='The image split to use, ``train``, ``test`` or ``val`` if mode="fine"\n        otherwise ``train``, ``train_extra`` or ``val``',
        required=True,
        default="train",
    )
    argument_parser.add_argument(
        "--mode",
        type=str,
        help="The quality mode to use, ``fine`` or ``coarse``",
        required=True,
        default="fine",
    )
    argument_parser.add_argument(
        "--target_type",
        type=str,
        help="Type of target to use, ``instance``, ``semantic``, ``polygon``\n        or ``color``. Can also be a list to output a tuple with all specified target types.",
        required=True,
        default="instance",
    )
    argument_parser.add_argument(
        "--transform",
        type=str,
        help="A function/transform that takes in a PIL image\n        and returns a transformed version. E.g, ``transforms.RandomCrop``",
    )
    argument_parser.add_argument(
        "--target_transform",
        type=str,
        help="A function/transform that takes in the\n        target and transforms it.",
    )
    argument_parser.add_argument(
        "--transforms",
        type=str,
        help="A function/transform that takes input sample and its target as entry\n        and returns a transformed version.",
    )
    argument_parser.add_argument(
        "--CityscapesClass",
        type=str,
        help="",
        required=True,
        default="```namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id', 'has_instances', 'ignore_in_eval', 'color'])```",
    )
    argument_parser.add_argument(
        "--classes",
        type=loads,
        help="",
        required=True,
        default="[CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)), CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)), CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)), CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)), CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)), CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)), CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)), CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)), CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)), CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)), CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)), CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)), CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)), CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)), CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)), CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)), CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)), CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)), CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)), CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)), CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)), CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)), CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)), CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)), CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)), CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)), CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)), CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)), CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)), CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)), CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)), CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)), CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)), CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)), CityscapesClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142))]",
    )
    return argument_parser


def CocoCaptionsConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "`MS Coco Captions <https://cocodataset.org/#captions-2015>`_ Dataset.\n\n\nExample:\n\n    .. code:: python\n\n        import torchvision.datasets as dset\n        import torchvision.transforms as transforms\n        cap = dset.CocoCaptions(root = 'dir where images are',\n                                annFile = 'json annotation file',\n                                transform=transforms.ToTensor())\n\n        print('Number of samples: ', len(cap))\n        img, target = cap[3] # load 4th sample\n\n        print(\"Image Size: \", img.size())\n        print(target)\n\n    Output: ::\n\n        Number of samples: 82783\n        Image Size: (3L, 427L, 640L)\n        [u'A plane emitting smoke stream flying over a mountain.',\n        u'A plane darts across a bright blue sky behind a mountain covered in snow',\n        u'A plane leaves a contrail above the snowy mountain top.',\n        u'A mountain that has a plane flying overheard in the distance.',\n        u'A mountain view with a plume of smoke in the background']"
    argument_parser.add_argument(
        "--root",
        type=str,
        help="Root directory where images are downloaded to.",
        required=True,
    )
    argument_parser.add_argument(
        "--annFile", type=str, help="Path to json annotation file.", required=True
    )
    argument_parser.add_argument(
        "--transform",
        type=str,
        help="A function/transform that  takes in an PIL image\n        and returns a transformed version. E.g, ``transforms.ToTensor``",
    )
    argument_parser.add_argument(
        "--target_transform",
        type=str,
        help="A function/transform that takes in the\n        target and transforms it.",
    )
    argument_parser.add_argument(
        "--transforms",
        type=str,
        help="A function/transform that takes input sample and its target as entry\n        and returns a transformed version.",
    )
    return argument_parser


def CocoDetectionConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = (
        "`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset."
    )
    argument_parser.add_argument(
        "--root",
        type=str,
        help="Root directory where images are downloaded to.",
        required=True,
    )
    argument_parser.add_argument(
        "--annFile", type=str, help="Path to json annotation file.", required=True
    )
    argument_parser.add_argument(
        "--transform",
        type=str,
        help="A function/transform that  takes in an PIL image\n        and returns a transformed version. E.g, ``transforms.ToTensor``",
    )
    argument_parser.add_argument(
        "--target_transform",
        type=str,
        help="A function/transform that takes in the\n        target and transforms it.",
    )
    argument_parser.add_argument(
        "--transforms",
        type=str,
        help="A function/transform that takes input sample and its target as entry\n        and returns a transformed version.",
    )
    return argument_parser


def DatasetFolderConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "A generic data loader where the samples are arranged in this way: ::\n\n    root/class_x/xxx.ext\n    root/class_x/xxy.ext\n    root/class_x/xxz.ext\n\n    root/class_y/123.ext\n    root/class_y/nsdf3.ext\n    root/class_y/asd932_.ext\n\n\n Attributes:\n    classes (list): List of the class names sorted alphabetically.\n    class_to_idx (dict): Dict with items (class_name, class_index).\n    samples (list): List of (sample path, class_index) tuples\n    targets (list): The class_index value for each image in the dataset"
    argument_parser.add_argument(
        "--root", type=str, help="Root directory path.", required=True
    )
    argument_parser.add_argument(
        "--loader",
        type=str,
        help="A function to load a sample given its path.",
        required=True,
    )
    argument_parser.add_argument(
        "--extensions",
        type=str,
        help="A list of allowed extensions.\n        both extensions and is_valid_file should not be passed.",
    )
    argument_parser.add_argument(
        "--transform",
        type=str,
        help="A function/transform that takes in\n        a sample and returns a transformed version.\n        E.g, ``transforms.RandomCrop`` for images.",
    )
    argument_parser.add_argument(
        "--target_transform",
        type=str,
        help="A function/transform that takes\n        in the target and transforms it.",
    )
    argument_parser.add_argument(
        "--is_valid_file",
        type=str,
        help="A function that takes path of a file\n        and check if the file is a valid file (used to check of corrupt files)\n        both extensions and is_valid_file should not be passed.",
    )
    return argument_parser


def EMNISTConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "`EMNIST <https://www.westernsydney.edu.au/bens/home/reproducible_research/emnist>`_ Dataset."
    argument_parser.add_argument(
        "--root",
        type=str,
        help="Root directory of dataset where ``EMNIST/processed/training.pt``\n        and  ``EMNIST/processed/test.pt`` exist.",
        required=True,
    )
    argument_parser.add_argument(
        "--split",
        type=str,
        help="The dataset has 6 different splits: ``byclass``, ``bymerge``,\n        ``balanced``, ``letters``, ``digits`` and ``mnist``. This argument specifies\n        which one to use.",
        required=True,
    )
    argument_parser.add_argument(
        "--train",
        type=bool,
        help="If True, creates dataset from ``training.pt``,\n        otherwise from ``test.pt``.",
    )
    argument_parser.add_argument(
        "--download",
        type=bool,
        help="If true, downloads the dataset from the internet and\n        puts it in root directory. If dataset is already downloaded, it is not\n        downloaded again.",
    )
    argument_parser.add_argument(
        "--transform",
        type=str,
        help="A function/transform that  takes in an PIL image\n        and returns a transformed version. E.g, ``transforms.RandomCrop``",
    )
    argument_parser.add_argument(
        "--target_transform",
        type=str,
        help="A function/transform that takes in the\n        target and transforms it.",
    )
    argument_parser.add_argument(
        "--splits",
        type=loads,
        help="",
        required=True,
        default="('byclass', 'bymerge', 'balanced', 'letters', 'digits', 'mnist')",
    )
    argument_parser.add_argument(
        "--_all_classes",
        type=str,
        help="",
        required=True,
        default="```set(list(string.digits + string.ascii_letters))```",
    )
    argument_parser.add_argument(
        "--md5",
        type=str,
        help="",
        required=True,
        default="58c8d27c78d21e728a6bc7b3cc06412e",
    )
    argument_parser.add_argument(
        "--classes_split_dict",
        type=loads,
        help="",
        required=True,
        default="{'byclass': list(_all_classes), 'bymerge': sorted(list(_all_classes - _merged_classes)), 'balanced': sorted(list(_all_classes - _merged_classes)), 'letters': list(string.ascii_lowercase), 'digits': list(string.digits), 'mnist': list(string.digits)}",
    )
    argument_parser.add_argument(
        "--url",
        type=str,
        help="",
        required=True,
        default="http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip",
    )
    argument_parser.add_argument(
        "--_merged_classes",
        type=str,
        help="",
        required=True,
        default="```set(['C', 'I', 'J', 'K', 'L', 'M', 'O', 'P', 'S', 'U', 'V', 'W', 'X', 'Y', 'Z'])```",
    )
    argument_parser.add_argument("--kwargs", type=loads, help="")
    return argument_parser


def FakeDataConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "A fake dataset that returns randomly generated images and returns them as PIL images"
    argument_parser.add_argument(
        "--size", type=int, help="Size of the dataset.", required=True, default=1000
    )
    argument_parser.add_argument(
        "--image_size",
        type=loads,
        help="Size if the returned images.",
        required=True,
        default="(3, 224, 224)",
    )
    argument_parser.add_argument(
        "--num_classes",
        type=int,
        help="Number of classes in the datset.",
        required=True,
        default=10,
    )
    argument_parser.add_argument(
        "--transform",
        type=str,
        help="A function/transform that  takes in an PIL image\n        and returns a transformed version. E.g, ``transforms.RandomCrop``",
    )
    argument_parser.add_argument(
        "--target_transform",
        type=str,
        help="A function/transform that takes in the\n        target and transforms it.",
    )
    argument_parser.add_argument(
        "--random_offset",
        type=int,
        help="Offsets the index-based random seed used to\n        generate each image.",
        required=True,
        default=0,
    )
    return argument_parser


def FashionMNISTConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = (
        "`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset."
    )
    argument_parser.add_argument(
        "--root",
        type=str,
        help="Root directory of dataset where ``FashionMNIST/processed/training.pt``\n        and  ``FashionMNIST/processed/test.pt`` exist.",
        required=True,
    )
    argument_parser.add_argument(
        "--train",
        type=bool,
        help="If True, creates dataset from ``training.pt``,\n        otherwise from ``test.pt``.",
        required=True,
        default=True,
    )
    argument_parser.add_argument(
        "--download",
        type=bool,
        help="If true, downloads the dataset from the internet and\n        puts it in root directory. If dataset is already downloaded, it is not\n        downloaded again.",
        required=True,
        default=False,
    )
    argument_parser.add_argument(
        "--transform",
        type=str,
        help="A function/transform that  takes in an PIL image\n        and returns a transformed version. E.g, ``transforms.RandomCrop``",
    )
    argument_parser.add_argument(
        "--target_transform",
        type=str,
        help="A function/transform that takes in the\n        target and transforms it.",
    )
    argument_parser.add_argument(
        "--resources",
        type=loads,
        help="",
        required=True,
        default="[('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz', '8d4fb7e6c68d591d4c3dfef9ec88bf0d'), ('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz', '25c81989df183df01b3e8a0aad5dffbe'), ('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz', 'bef4ecab320f06d8554ea6380940ec79'), ('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz', 'bb300cfdad3c16e7a12a480ee83cd310')]",
    )
    argument_parser.add_argument(
        "--classes",
        type=loads,
        help="",
        required=True,
        default="['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']",
    )
    return argument_parser


def Flickr30kConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "`Flickr30k Entities <http://web.engr.illinois.edu/~bplumme2/Flickr30kEntities/>`_ Dataset."
    argument_parser.add_argument(
        "--root",
        type=str,
        help="Root directory where images are downloaded to.",
        required=True,
    )
    argument_parser.add_argument(
        "--ann_file", type=str, help="Path to annotation file.", required=True
    )
    argument_parser.add_argument(
        "--transform",
        type=str,
        help="A function/transform that takes in a PIL image\n        and returns a transformed version. E.g, ``transforms.ToTensor``",
    )
    argument_parser.add_argument(
        "--target_transform",
        type=str,
        help="A function/transform that takes in the\n        target and transforms it.",
    )
    return argument_parser


def Flickr8kConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "`Flickr8k Entities <http://hockenmaier.cs.illinois.edu/8k-pictures.html>`_ Dataset."
    argument_parser.add_argument(
        "--root",
        type=str,
        help="Root directory where images are downloaded to.",
        required=True,
    )
    argument_parser.add_argument(
        "--ann_file", type=str, help="Path to annotation file.", required=True
    )
    argument_parser.add_argument(
        "--transform",
        type=str,
        help="A function/transform that takes in a PIL image\n        and returns a transformed version. E.g, ``transforms.ToTensor``",
    )
    argument_parser.add_argument(
        "--target_transform",
        type=str,
        help="A function/transform that takes in the\n        target and transforms it.",
    )
    return argument_parser


def HMDB51Config(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser, video (Tensor[T, H, W, C]): the `T` video frames
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "`HMDB51 <http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/>`_\ndataset.\n\nHMDB51 is an action recognition video dataset.\nThis dataset consider every video as a collection of video clips of fixed size, specified\nby ``frames_per_clip``, where the step in frames between each clip is given by\n``step_between_clips``.\n\nTo give an example, for 2 videos with 10 and 15 frames respectively, if ``frames_per_clip=5``\nand ``step_between_clips=5``, the dataset size will be (2 + 3) = 5, where the first two\nelements will come from video 1, and the next three elements from video 2.\nNote that we drop clips which do not have exactly ``frames_per_clip`` elements, so not all\nframes in a video might be present.\n\nInternally, it uses a VideoClips object to handle clip creation."
    argument_parser.add_argument(
        "--root",
        type=int,
        help="Root directory of the HMDB51 Dataset.",
        required=True,
        default=1,
    )
    argument_parser.add_argument(
        "--annotation_path",
        type=str,
        help="Path to the folder containing the split files.",
    )
    argument_parser.add_argument(
        "--frames_per_clip",
        type=int,
        help="Number of frames in a clip.",
        required=True,
        default=1,
    )
    argument_parser.add_argument(
        "--step_between_clips",
        type=int,
        help="Number of frames between each clip.",
        required=True,
        default=1,
    )
    argument_parser.add_argument(
        "--fold",
        type=int,
        help="Which fold to use. Should be between 1 and 3.",
        default=1,
    )
    argument_parser.add_argument(
        "--train",
        type=bool,
        help="If ``True``, creates a dataset from the train split,\n        otherwise from the ``test`` split.",
        default=True,
    )
    argument_parser.add_argument(
        "--transform",
        type=int,
        help="A function/transform that takes in a TxHxWxC video\n        and returns a transformed version.",
        default=0,
    )
    argument_parser.add_argument(
        "--splits",
        type=loads,
        help="",
        required=True,
        default="{'url': 'http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar', 'md5': '15e67781e70dcfbdce2d7dbb9b3344b5'}",
    )
    argument_parser.add_argument(
        "--TRAIN_TAG", type=int, help="", required=True, default=1
    )
    argument_parser.add_argument(
        "--data_url",
        type=str,
        help="",
        required=True,
        default="http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar",
    )
    argument_parser.add_argument(
        "--TEST_TAG", type=int, help="", required=True, default=2
    )
    argument_parser.add_argument(
        "--_precomputed_metadata", help=None, required=True, default=0
    )
    argument_parser.add_argument("--frame_rate", help=None)
    argument_parser.add_argument("--num_workers", help=None, required=True, default=0)
    argument_parser.add_argument("--_video_width", help=None, required=True, default=0)
    argument_parser.add_argument("--_video_height", help=None, required=True)
    argument_parser.add_argument("--_video_min_dimension", help=None, required=True)
    argument_parser.add_argument("--_audio_samples", help=None, required=True)
    return argument_parser


def ImageFolderConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "A generic data loader where the images are arranged in this way: ::\n\n    root/dog/xxx.png\n    root/dog/xxy.png\n    root/dog/xxz.png\n\n    root/cat/123.png\n    root/cat/nsdf3.png\n    root/cat/asd932_.png\n\n\n Attributes:\n    classes (list): List of the class names sorted alphabetically.\n    class_to_idx (dict): Dict with items (class_name, class_index).\n    imgs (list): List of (image path, class_index) tuples"
    argument_parser.add_argument(
        "--root", type=str, help="Root directory path.", required=True
    )
    argument_parser.add_argument(
        "--transform",
        type=str,
        help="A function/transform that  takes in an PIL image\n        and returns a transformed version. E.g, ``transforms.RandomCrop``",
    )
    argument_parser.add_argument(
        "--target_transform",
        type=str,
        help="A function/transform that takes in the\n        target and transforms it.",
        default="default_loader",
    )
    argument_parser.add_argument(
        "--loader",
        type=pickle.loads,
        help="A function to load an image given its path.",
        required=True,
        default="b'\\x80\\x04\\x952\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x8c\\x1btorchvision.datasets.folder\\x94\\x8c\\x0edefault_loader\\x94\\x93\\x94.'",
    )
    argument_parser.add_argument(
        "--is_valid_file",
        type=str,
        help="A function that takes path of an Image file\n        and check if the file is a valid file (used to check of corrupt files)",
    )
    return argument_parser


def ImageNetConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "`ImageNet <http://image-net.org/>`_ 2012 Classification Dataset.\n\n\n Attributes:\n    classes (list): List of the class name tuples.\n    class_to_idx (dict): Dict with items (class_name, class_index).\n    wnids (list): List of the WordNet IDs.\n    wnid_to_idx (dict): Dict with items (wordnet_id, class_index).\n    imgs (list): List of (image path, class_index) tuples\n    targets (list): The class_index value for each image in the dataset"
    argument_parser.add_argument(
        "--root",
        type=str,
        help="Root directory of the ImageNet Dataset.",
        required=True,
        default="train",
    )
    argument_parser.add_argument(
        "--split",
        type=str,
        help="The dataset split, supports ``train``, or ``val``.",
        required=True,
        default="train",
    )
    argument_parser.add_argument(
        "--transform",
        type=str,
        help="A function/transform that  takes in an PIL image\n        and returns a transformed version. E.g, ``transforms.RandomCrop``",
    )
    argument_parser.add_argument(
        "--target_transform",
        type=str,
        help="A function/transform that takes in the\n        target and transforms it.",
    )
    argument_parser.add_argument(
        "--loader", type=str, help="A function to load an image given its path."
    )
    argument_parser.add_argument("--download", type=str, help=None)
    argument_parser.add_argument("--kwargs", type=loads, help="")
    return argument_parser


def KMNISTConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = (
        "`Kuzushiji-MNIST <https://github.com/rois-codh/kmnist>`_ Dataset."
    )
    argument_parser.add_argument(
        "--root",
        type=str,
        help="Root directory of dataset where ``KMNIST/processed/training.pt``\n        and  ``KMNIST/processed/test.pt`` exist.",
        required=True,
    )
    argument_parser.add_argument(
        "--train",
        type=bool,
        help="If True, creates dataset from ``training.pt``,\n        otherwise from ``test.pt``.",
        required=True,
        default=True,
    )
    argument_parser.add_argument(
        "--download",
        type=bool,
        help="If true, downloads the dataset from the internet and\n        puts it in root directory. If dataset is already downloaded, it is not\n        downloaded again.",
        required=True,
        default=False,
    )
    argument_parser.add_argument(
        "--transform",
        type=str,
        help="A function/transform that  takes in an PIL image\n        and returns a transformed version. E.g, ``transforms.RandomCrop``",
    )
    argument_parser.add_argument(
        "--target_transform",
        type=str,
        help="A function/transform that takes in the\n        target and transforms it.",
    )
    argument_parser.add_argument(
        "--resources",
        type=loads,
        help="",
        required=True,
        default="[('http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz', 'bdb82020997e1d708af4cf47b453dcf7'), ('http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz', 'e144d726b3acfaa3e44228e80efcd344'), ('http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz', '5c965bf0a639b31b8f53240b1b52f4d7'), ('http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz', '7320c461ea6c1c855c0b718fb2a4b134')]",
    )
    argument_parser.add_argument(
        "--classes",
        type=loads,
        help="",
        required=True,
        default="['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo']",
    )
    return argument_parser


def Kinetics400Config(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser, video (Tensor[T, H, W, C]): the `T` video frames
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "`Kinetics-400 <https://deepmind.com/research/open-source/open-source-datasets/kinetics/>`_\ndataset.\n\nKinetics-400 is an action recognition video dataset.\nThis dataset consider every video as a collection of video clips of fixed size, specified\nby ``frames_per_clip``, where the step in frames between each clip is given by\n``step_between_clips``.\n\nTo give an example, for 2 videos with 10 and 15 frames respectively, if ``frames_per_clip=5``\nand ``step_between_clips=5``, the dataset size will be (2 + 3) = 5, where the first two\nelements will come from video 1, and the next three elements from video 2.\nNote that we drop clips which do not have exactly ``frames_per_clip`` elements, so not all\nframes in a video might be present.\n\nInternally, it uses a VideoClips object to handle clip creation."
    argument_parser.add_argument(
        "--root",
        type=int,
        help="Root directory of the Kinetics-400 Dataset.",
        required=True,
        default=1,
    )
    argument_parser.add_argument(
        "--frames_per_clip", type=int, help="number of frames in a clip"
    )
    argument_parser.add_argument(
        "--step_between_clips",
        type=int,
        help="number of frames between each clip",
        required=True,
        default=1,
    )
    argument_parser.add_argument(
        "--transform",
        type=int,
        help="A function/transform that  takes in a TxHxWxC video\n        and returns a transformed version.",
        default=1,
    )
    argument_parser.add_argument(
        "--_precomputed_metadata", help=None, required=True, default=0
    )
    argument_parser.add_argument("--num_workers", help=None, required=True, default=0)
    argument_parser.add_argument("--frame_rate", help=None)
    argument_parser.add_argument(
        "--_video_min_dimension", help=None, required=True, default=0
    )
    argument_parser.add_argument("--_video_width", help=None, required=True, default=0)
    argument_parser.add_argument("--_video_height", help=None, required=True, default=0)
    argument_parser.add_argument("--_audio_channels", help=None, required=True)
    argument_parser.add_argument("--extensions", help=None)
    argument_parser.add_argument("--_audio_samples", help=None, required=True)
    return argument_parser


def LSUNConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "`LSUN <https://www.yf.io/p/lsun>`_ dataset."
    argument_parser.add_argument(
        "--root",
        type=str,
        help="Root directory for the database files.",
        required=True,
        default="train",
    )
    argument_parser.add_argument(
        "--classes",
        type=str,
        help="One of {'train', 'val', 'test'} or a list of\n        categories to load. e,g. ['bedroom_train', 'church_outdoor_train'].",
        required=True,
        default="train",
    )
    argument_parser.add_argument(
        "--transform",
        type=str,
        help="A function/transform that  takes in an PIL image\n        and returns a transformed version. E.g, ``transforms.RandomCrop``",
    )
    argument_parser.add_argument(
        "--target_transform",
        type=str,
        help="A function/transform that takes in the\n        target and transforms it.",
    )
    return argument_parser


def LSUNClassConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "An abstract class representing a :class:`Dataset`.\n\nAll datasets that represent a map from keys to data samples should subclass\nit. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a\ndata sample for a given key. Subclasses could also optionally overwrite\n:meth:`__len__`, which is expected to return the size of the dataset by many\n:class:`~torch.utils.data.Sampler` implementations and the default options\nof :class:`~torch.utils.data.DataLoader`.\n\n.. note::\n  :class:`~torch.utils.data.DataLoader` by default constructs a index\n  sampler that yields integral indices.  To make it work with a map-style\n  dataset with non-integral indices/keys, a custom sampler must be provided."
    argument_parser.add_argument("--root", type=str, help=None, required=True)
    argument_parser.add_argument("--transform", type=str, help=None)
    argument_parser.add_argument("--target_transform", type=str, help=None)
    return argument_parser


def MNISTConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = (
        "`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset."
    )
    argument_parser.add_argument(
        "--root",
        type=bool,
        help="Root directory of dataset where ``MNIST/processed/training.pt``\n        and  ``MNIST/processed/test.pt`` exist.",
        required=True,
        default=True,
    )
    argument_parser.add_argument(
        "--train",
        type=bool,
        help="If True, creates dataset from ``training.pt``,\n        otherwise from ``test.pt``.",
        required=True,
        default=True,
    )
    argument_parser.add_argument(
        "--download",
        type=bool,
        help="If true, downloads the dataset from the internet and\n        puts it in root directory. If dataset is already downloaded, it is not\n        downloaded again.",
        required=True,
        default=False,
    )
    argument_parser.add_argument(
        "--transform",
        type=str,
        help="A function/transform that  takes in an PIL image\n        and returns a transformed version. E.g, ``transforms.RandomCrop``",
    )
    argument_parser.add_argument(
        "--target_transform",
        type=bool,
        help="A function/transform that takes in the\n        target and transforms it.",
        default=False,
    )
    argument_parser.add_argument(
        "--resources",
        type=loads,
        help="",
        required=True,
        default="[('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873'), ('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'), ('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3'), ('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c')]",
    )
    argument_parser.add_argument(
        "--classes",
        type=loads,
        help="",
        required=True,
        default="['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four', '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']",
    )
    argument_parser.add_argument(
        "--training_file", type=str, help="", required=True, default="training.pt"
    )
    argument_parser.add_argument(
        "--test_file", type=str, help="", required=True, default="test.pt"
    )
    return argument_parser


def OmniglotConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = (
        "`Omniglot <https://github.com/brendenlake/omniglot>`_ Dataset."
    )
    argument_parser.add_argument(
        "--root",
        type=bool,
        help="Root directory of dataset where directory\n        ``omniglot-py`` exists.",
        required=True,
        default=True,
    )
    argument_parser.add_argument(
        "--background",
        type=bool,
        help='If True, creates dataset from the "background" set, otherwise\n        creates from the "evaluation" set. This terminology is defined by the authors.',
        required=True,
        default=True,
    )
    argument_parser.add_argument(
        "--transform",
        type=str,
        help="A function/transform that  takes in an PIL image\n        and returns a transformed version. E.g, ``transforms.RandomCrop``",
    )
    argument_parser.add_argument(
        "--target_transform",
        type=bool,
        help="A function/transform that takes in the\n        target and transforms it.",
        default=False,
    )
    argument_parser.add_argument(
        "--download",
        type=bool,
        help="If true, downloads the dataset zip files from the internet and\n        puts it in root directory. If the zip files are already downloaded, they are not\n        downloaded again.",
        required=True,
        default=False,
    )
    argument_parser.add_argument(
        "--folder", type=str, help="", required=True, default="omniglot-py"
    )
    argument_parser.add_argument(
        "--zips_md5",
        type=loads,
        help="",
        required=True,
        default="{'images_background': '68d2efa1b9178cc56df9314c21c6e718', 'images_evaluation': '6b91aef0f799c5bb55b94e3f2daec811'}",
    )
    argument_parser.add_argument(
        "--download_url_prefix",
        type=str,
        help="",
        required=True,
        default="https://github.com/brendenlake/omniglot/raw/master/python",
    )
    return argument_parser


def PhotoTourConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "`Learning Local Image Descriptors Data <http://phototour.cs.washington.edu/patches/default.htm>`_ Dataset."
    argument_parser.add_argument(
        "--root",
        type=bool,
        help="Root directory where images are.",
        required=True,
        default=True,
    )
    argument_parser.add_argument(
        "--name", type=str, help="Name of the dataset to load.", required=True
    )
    argument_parser.add_argument(
        "--transform",
        type=str,
        help="A function/transform that  takes in an PIL image\n        and returns a transformed version.",
    )
    argument_parser.add_argument(
        "--download",
        type=bool,
        help="If true, downloads the dataset from the internet and\n        puts it in root directory. If dataset is already downloaded, it is not\n        downloaded again.",
        required=True,
        default=False,
    )
    argument_parser.add_argument(
        "--image_ext", type=str, help="", required=True, default="bmp"
    )
    argument_parser.add_argument(
        "--stds",
        type=loads,
        help="",
        required=True,
        default="{'notredame': 0.1864, 'yosemite': 0.1818, 'liberty': 0.2019, 'notredame_harris': 0.1864, 'yosemite_harris': 0.1818, 'liberty_harris': 0.2019}",
    )
    argument_parser.add_argument(
        "--info_file", type=str, help="", required=True, default="info.txt"
    )
    argument_parser.add_argument(
        "--matches_files",
        type=str,
        help="",
        required=True,
        default="m50_100000_100000_0.txt",
    )
    argument_parser.add_argument(
        "--means",
        type=loads,
        help="",
        required=True,
        default="{'notredame': 0.4854, 'yosemite': 0.4844, 'liberty': 0.4437, 'notredame_harris': 0.4854, 'yosemite_harris': 0.4844, 'liberty_harris': 0.4437}",
    )
    argument_parser.add_argument(
        "--lens",
        type=loads,
        help="",
        required=True,
        default="{'notredame': 468159, 'yosemite': 633587, 'liberty': 450092, 'liberty_harris': 379587, 'yosemite_harris': 450912, 'notredame_harris': 325295}",
    )
    argument_parser.add_argument(
        "--urls",
        type=loads,
        help="",
        required=True,
        default="{'notredame_harris': ['http://matthewalunbrown.com/patchdata/notredame_harris.zip', 'notredame_harris.zip', '69f8c90f78e171349abdf0307afefe4d'], 'yosemite_harris': ['http://matthewalunbrown.com/patchdata/yosemite_harris.zip', 'yosemite_harris.zip', 'a73253d1c6fbd3ba2613c45065c00d46'], 'liberty_harris': ['http://matthewalunbrown.com/patchdata/liberty_harris.zip', 'liberty_harris.zip', 'c731fcfb3abb4091110d0ae8c7ba182c'], 'notredame': ['http://icvl.ee.ic.ac.uk/vbalnt/notredame.zip', 'notredame.zip', '509eda8535847b8c0a90bbb210c83484'], 'yosemite': ['http://icvl.ee.ic.ac.uk/vbalnt/yosemite.zip', 'yosemite.zip', '533b2e8eb7ede31be40abc317b2fd4f0'], 'liberty': ['http://icvl.ee.ic.ac.uk/vbalnt/liberty.zip', 'liberty.zip', 'fdd9152f138ea5ef2091746689176414']}",
    )
    argument_parser.add_argument(
        "--train", type=bool, help=None, required=True, default=False
    )
    return argument_parser


def Places365Config(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "`Places365 <http://places2.csail.mit.edu/index.html>`_ classification dataset.\n\n\n Attributes:\n    classes (list): List of the class names.\n    class_to_idx (dict): Dict with items (class_name, class_index).\n    imgs (list): List of (image path, class_index) tuples\n    targets (list): The class_index value for each image in the dataset\n\nRaises:\n    RuntimeError: If ``download is False`` and the meta files, i. e. the devkit, are not present or corrupted.\n    RuntimeError: If ``download is True`` and the image archive is already extracted."
    argument_parser.add_argument(
        "--root",
        type=str,
        help="Root directory of the Places365 dataset.",
        required=True,
        default="train-standard",
    )
    argument_parser.add_argument(
        "--split",
        type=str,
        help="The dataset split. Can be one of ``train-standard`` (default), ``train-challendge``,\n        ``val``.",
        required=True,
        default="train-standard",
    )
    argument_parser.add_argument(
        "--small",
        type=bool,
        help="If ``True``, uses the small images, i. e. resized to 256 x 256 pixels, instead of the\n        high resolution ones.",
        required=True,
        default=False,
    )
    argument_parser.add_argument(
        "--download",
        type=bool,
        help="If ``True``, downloads the dataset components and places them in ``root``. Already\n        downloaded archives are not downloaded again.",
        required=True,
        default=False,
    )
    argument_parser.add_argument(
        "--transform",
        type=str,
        help="A function/transform that  takes in an PIL image\n        and returns a transformed version. E.g, ``transforms.RandomCrop``",
    )
    argument_parser.add_argument(
        "--target_transform",
        type=str,
        help="A function/transform that takes in the\n        target and transforms it.",
        default="default_loader",
    )
    argument_parser.add_argument(
        "--loader",
        type=pickle.loads,
        help="A function to load an image given its path.",
        required=True,
        default="b'\\x80\\x04\\x952\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x8c\\x1btorchvision.datasets.folder\\x94\\x8c\\x0edefault_loader\\x94\\x93\\x94.'",
    )
    argument_parser.add_argument(
        "--_CATEGORIES_META",
        type=loads,
        help="",
        required=True,
        default="('categories_places365.txt', '06c963b85866bd0649f97cb43dd16673')",
    )
    argument_parser.add_argument(
        "--_BASE_URL",
        type=str,
        help="",
        required=True,
        default="http://data.csail.mit.edu/places/places365/",
    )
    argument_parser.add_argument(
        "--_FILE_LIST_META",
        type=loads,
        help="",
        required=True,
        default="{'train-standard': ('places365_train_standard.txt', '30f37515461640559006b8329efbed1a'), 'train-challenge': ('places365_train_challenge.txt', 'b2931dc997b8c33c27e7329c073a6b57'), 'val': ('places365_val.txt', 'e9f2fd57bfd9d07630173f4e8708e4b1')}",
    )
    argument_parser.add_argument(
        "--_DEVKIT_META",
        type=loads,
        help="",
        required=True,
        default="{'standard': ('filelist_places365-standard.tar', '35a0585fee1fa656440f3ab298f8479c'), 'challenge': ('filelist_places365-challenge.tar', '70a8307e459c3de41690a7c76c931734')}",
    )
    argument_parser.add_argument(
        "--_IMAGES_META",
        type=loads,
        help="",
        required=True,
        default="{('train-standard', False): ('train_large_places365standard.tar', '67e186b496a84c929568076ed01a8aa1'), ('train-challenge', False): ('train_large_places365challenge.tar', '605f18e68e510c82b958664ea134545f'), ('val', False): ('val_large.tar', '9b71c4993ad89d2d8bcbdc4aef38042f'), ('train-standard', True): ('train_256_places365standard.tar', '53ca1c756c3d1e7809517cc47c5561c5'), ('train-challenge', True): ('train_256_places365challenge.tar', '741915038a5e3471ec7332404dfb64ef'), ('val', True): ('val_256.tar', 'e27b17d8d44f4af9a78502beb927f808')}",
    )
    argument_parser.add_argument(
        "--_SPLITS",
        type=loads,
        help="",
        required=True,
        default="('train-standard', 'train-challenge', 'val')",
    )
    return argument_parser


def QMNISTConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = (
        "`QMNIST <https://github.com/facebookresearch/qmnist>`_ Dataset."
    )
    argument_parser.add_argument(
        "--root",
        type=str,
        help="Root directory of dataset whose ``processed''\n        subdir contains torch binary files with the datasets.",
        required=True,
    )
    argument_parser.add_argument(
        "--what",
        type=bool,
        help="Can be 'train', 'test', 'test10k',\n        'test50k', or 'nist' for respectively the mnist compatible\n        training set, the 60k qmnist testing set, the 10k qmnist\n        examples that match the mnist testing set, the 50k\n        remaining qmnist testing examples, or all the nist\n        digits. The default is to select 'train' or 'test'\n        according to the compatibility argument 'train'.",
        default=True,
    )
    argument_parser.add_argument(
        "--compat",
        type=bool,
        help="A boolean that says whether the target\n        for each example is class number (for compatibility with\n        the MNIST dataloader) or a torch vector containing the\n        full qmnist information. Default=True.",
        required=True,
        default=True,
    )
    argument_parser.add_argument(
        "--download",
        type=bool,
        help="If true, downloads the dataset from\n        the internet and puts it in root directory. If dataset is\n        already downloaded, it is not downloaded again.",
    )
    argument_parser.add_argument(
        "--transform",
        type=str,
        help="A function/transform that\n        takes in an PIL image and returns a transformed\n        version. E.g, ``transforms.RandomCrop``",
    )
    argument_parser.add_argument(
        "--target_transform",
        type=str,
        help="A function/transform\n        that takes in the target and transforms it.",
    )
    argument_parser.add_argument(
        "--train",
        type=bool,
        help="When argument 'what' is\n        not specified, this boolean decides whether to load the\n        training set ot the testing set. ",
        required=True,
        default=True,
    )
    argument_parser.add_argument(
        "--resources",
        type=str,
        action="append",
        help="",
        required=True,
        default="{'train': [('https://raw.githubusercontent.com/facebookresearch/qmnist/master/qmnist-train-images-idx3-ubyte.gz', 'ed72d4157d28c017586c42bc6afe6370'), ('https://raw.githubusercontent.com/facebookresearch/qmnist/master/qmnist-train-labels-idx2-int.gz', '0058f8dd561b90ffdd0f734c6a30e5e4')], 'test': [('https://raw.githubusercontent.com/facebookresearch/qmnist/master/qmnist-test-images-idx3-ubyte.gz', '1394631089c404de565df7b7aeaf9412'), ('https://raw.githubusercontent.com/facebookresearch/qmnist/master/qmnist-test-labels-idx2-int.gz', '5b5b05890a5e13444e108efe57b788aa')], 'nist': [('https://raw.githubusercontent.com/facebookresearch/qmnist/master/xnist-images-idx3-ubyte.xz', '7f124b3b8ab81486c9d8c2749c17f834'), ('https://raw.githubusercontent.com/facebookresearch/qmnist/master/xnist-labels-idx2-int.xz', '5ed0e788978e45d4a8bd4b7caec3d79d')]}",
    )
    argument_parser.add_argument(
        "--classes",
        type=loads,
        help="",
        required=True,
        default="['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four', '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']",
    )
    argument_parser.add_argument(
        "--subsets",
        type=loads,
        help="",
        required=True,
        default="{'train': 'train', 'test': 'test', 'test10k': 'test', 'test50k': 'test', 'nist': 'nist'}",
    )
    argument_parser.add_argument("--kwargs", type=loads, help="")
    return argument_parser


def SBDatasetConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "`Semantic Boundaries Dataset <http://home.bharathh.info/pubs/codes/SBD/download.html>`_\n\nThe SBD currently contains annotations from 11355 images taken from the PASCAL VOC 2011 dataset.\n\n.. note ::\n\n    Please note that the train and val splits included with this dataset are different from\n    the splits in the PASCAL VOC dataset. In particular some \"train\" images might be part of\n    VOC2012 val.\n    If you are interested in testing on VOC 2012 val, then use `image_set='train_noval'`,\n    which excludes all val images.\n\n.. warning::\n\n    This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format."
    argument_parser.add_argument(
        "--root",
        type=str,
        help="Root directory of the Semantic Boundaries Dataset",
        required=True,
        default="train",
    )
    argument_parser.add_argument(
        "--image_set",
        type=str,
        help="Select the image_set to use, ``train``, ``val`` or ``train_noval``.\n        Image set ``train_noval`` excludes VOC 2012 val images.",
        required=True,
        default="train",
    )
    argument_parser.add_argument(
        "--mode",
        type=str,
        help="Select target type. Possible values 'boundaries' or 'segmentation'.\n        In case of 'boundaries', the target is an array of shape `[num_classes, H, W]`,\n        where `num_classes=20`.",
        required=True,
        default="boundaries",
    )
    argument_parser.add_argument(
        "--download",
        type=bool,
        help="If true, downloads the dataset from the internet and\n        puts it in root directory. If dataset is already downloaded, it is not\n        downloaded again.",
        required=True,
        default=False,
    )
    argument_parser.add_argument(
        "--transforms",
        type=str,
        help="A function/transform that takes input sample and its target as entry\n        and returns a transformed version. Input sample is PIL image and target is a numpy array\n        if `mode='boundaries'` or PIL image if `mode='segmentation'`.",
    )
    argument_parser.add_argument(
        "--filename", type=str, help="", required=True, default="benchmark.tgz"
    )
    argument_parser.add_argument(
        "--md5",
        type=str,
        help="",
        required=True,
        default="82b4d87ceb2ed10f6038a1cba92111cb",
    )
    argument_parser.add_argument(
        "--voc_train_url",
        type=str,
        help="",
        required=True,
        default="http://home.bharathh.info/pubs/codes/SBD/train_noval.txt",
    )
    argument_parser.add_argument(
        "--voc_split_md5",
        type=str,
        help="",
        required=True,
        default="79bff800c5f0b1ec6b21080a3c066722",
    )
    argument_parser.add_argument(
        "--voc_split_filename",
        type=str,
        help="",
        required=True,
        default="train_noval.txt",
    )
    argument_parser.add_argument(
        "--url",
        type=str,
        help="",
        required=True,
        default="http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz",
    )
    return argument_parser


def SBUConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "`SBU Captioned Photo <http://www.cs.virginia.edu/~vicente/sbucaptions/>`_ Dataset."
    argument_parser.add_argument(
        "--root",
        type=str,
        help="Root directory of dataset where tarball\n        ``SBUCaptionedPhotoDataset.tar.gz`` exists.",
        required=True,
    )
    argument_parser.add_argument(
        "--transform",
        type=str,
        help="A function/transform that takes in a PIL image\n        and returns a transformed version. E.g, ``transforms.RandomCrop``",
    )
    argument_parser.add_argument(
        "--target_transform",
        type=bool,
        help="A function/transform that takes in the\n        target and transforms it.",
        default=True,
    )
    argument_parser.add_argument(
        "--download",
        type=bool,
        help="If True, downloads the dataset from the internet and\n        puts it in root directory. If dataset is already downloaded, it is not\n        downloaded again.",
        required=True,
        default=True,
    )
    argument_parser.add_argument(
        "--md5_checksum",
        type=str,
        help="",
        required=True,
        default="9aec147b3488753cf758b4d493422285",
    )
    argument_parser.add_argument(
        "--filename",
        type=str,
        help="",
        required=True,
        default="SBUCaptionedPhotoDataset.tar.gz",
    )
    argument_parser.add_argument(
        "--url",
        type=str,
        help="",
        required=True,
        default="http://www.cs.virginia.edu/~vicente/sbucaptions/SBUCaptionedPhotoDataset.tar.gz",
    )
    return argument_parser


def SEMEIONConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "`SEMEION <http://archive.ics.uci.edu/ml/datasets/semeion+handwritten+digit>`_ Dataset."
    argument_parser.add_argument(
        "--root",
        type=str,
        help="Root directory of dataset where directory\n        ``semeion.py`` exists.",
        required=True,
    )
    argument_parser.add_argument(
        "--transform",
        type=str,
        help="A function/transform that  takes in an PIL image\n        and returns a transformed version. E.g, ``transforms.RandomCrop``",
    )
    argument_parser.add_argument(
        "--target_transform",
        type=bool,
        help="A function/transform that takes in the\n        target and transforms it.",
        default=True,
    )
    argument_parser.add_argument(
        "--download",
        type=bool,
        help="If true, downloads the dataset from the internet and\n        puts it in root directory. If dataset is already downloaded, it is not\n        downloaded again.",
        required=True,
        default=True,
    )
    argument_parser.add_argument(
        "--md5_checksum",
        type=str,
        help="",
        required=True,
        default="cb545d371d2ce14ec121470795a77432",
    )
    argument_parser.add_argument(
        "--filename", type=str, help="", required=True, default="semeion.data"
    )
    argument_parser.add_argument(
        "--url",
        type=str,
        help="",
        required=True,
        default="http://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data",
    )
    return argument_parser


def STL10Config(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = (
        "`STL10 <https://cs.stanford.edu/~acoates/stl10/>`_ Dataset."
    )
    argument_parser.add_argument(
        "--root",
        type=str,
        help="Root directory of dataset where directory\n        ``stl10_binary`` exists.",
        required=True,
        default="train",
    )
    argument_parser.add_argument(
        "--split",
        type=str,
        help="One of {'train', 'test', 'unlabeled', 'train+unlabeled'}.\n        Accordingly dataset is selected.",
        required=True,
        default="train",
    )
    argument_parser.add_argument(
        "--folds",
        type=int,
        help="One of {0-9} or None.\n        For training, loads one of the 10 pre-defined folds of 1k samples for the\n         standard evaluation procedure. If no value is passed, loads the 5k samples.",
    )
    argument_parser.add_argument(
        "--transform",
        type=str,
        help="A function/transform that  takes in an PIL image\n        and returns a transformed version. E.g, ``transforms.RandomCrop``",
    )
    argument_parser.add_argument(
        "--target_transform",
        type=bool,
        help="A function/transform that takes in the\n        target and transforms it.",
        default=False,
    )
    argument_parser.add_argument(
        "--download",
        type=bool,
        help="If true, downloads the dataset from the internet and\n        puts it in root directory. If dataset is already downloaded, it is not\n        downloaded again.",
        required=True,
        default=False,
    )
    argument_parser.add_argument(
        "--splits",
        type=loads,
        help="",
        required=True,
        default="('train', 'train+unlabeled', 'unlabeled', 'test')",
    )
    argument_parser.add_argument(
        "--filename", type=str, help="", required=True, default="stl10_binary.tar.gz"
    )
    argument_parser.add_argument(
        "--test_list",
        type=loads,
        help="",
        required=True,
        default="[['test_X.bin', '7f263ba9f9e0b06b93213547f721ac82'], ['test_y.bin', '36f9794fa4beb8a2c72628de14fa638e']]",
    )
    argument_parser.add_argument(
        "--tgz_md5",
        type=str,
        help="",
        required=True,
        default="91f7769df0f17e558f3565bffb0c7dfb",
    )
    argument_parser.add_argument(
        "--train_list",
        type=loads,
        help="",
        required=True,
        default="[['train_X.bin', '918c2871b30a85fa023e0c44e0bee87f'], ['train_y.bin', '5a34089d4802c674881badbb80307741'], ['unlabeled_X.bin', '5242ba1fed5e4be9e1e742405eb56ca4']]",
    )
    argument_parser.add_argument(
        "--folds_list_file",
        type=str,
        help="",
        required=True,
        default="fold_indices.txt",
    )
    argument_parser.add_argument(
        "--url",
        type=str,
        help="",
        required=True,
        default="http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz",
    )
    argument_parser.add_argument(
        "--class_names_file",
        type=str,
        help="",
        required=True,
        default="class_names.txt",
    )
    argument_parser.add_argument(
        "--base_folder", type=str, help="", required=True, default="stl10_binary"
    )
    return argument_parser


def SVHNConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "`SVHN <http://ufldl.stanford.edu/housenumbers/>`_ Dataset.\nNote: The SVHN dataset assigns the label `10` to the digit `0`. However, in this Dataset,\nwe assign the label `0` to the digit `0` to be compatible with PyTorch loss functions which\nexpect the class labels to be in the range `[0, C-1]`\n\n.. warning::\n\n    This class needs `scipy <https://docs.scipy.org/doc/>`_ to load data from `.mat` format."
    argument_parser.add_argument(
        "--root",
        type=str,
        help="Root directory of dataset where directory\n        ``SVHN`` exists.",
        required=True,
        default="train",
    )
    argument_parser.add_argument(
        "--split",
        type=str,
        help="One of {'train', 'test', 'extra'}.\n        Accordingly dataset is selected. 'extra' is Extra training set.",
        required=True,
        default="train",
    )
    argument_parser.add_argument(
        "--transform",
        type=str,
        help="A function/transform that  takes in an PIL image\n        and returns a transformed version. E.g, ``transforms.RandomCrop``",
    )
    argument_parser.add_argument(
        "--target_transform",
        type=bool,
        help="A function/transform that takes in the\n        target and transforms it.",
        default=False,
    )
    argument_parser.add_argument(
        "--download",
        type=bool,
        help="If true, downloads the dataset from the internet and\n        puts it in root directory. If dataset is already downloaded, it is not\n        downloaded again.",
        required=True,
        default=False,
    )
    argument_parser.add_argument(
        "--split_list",
        type=loads,
        help="",
        required=True,
        default="{'train': ['http://ufldl.stanford.edu/housenumbers/train_32x32.mat', 'train_32x32.mat', 'e26dedcc434d2e4c54c9b2d4a06d8373'], 'test': ['http://ufldl.stanford.edu/housenumbers/test_32x32.mat', 'test_32x32.mat', 'eb5a983be6a315427106f1b164d9cef3'], 'extra': ['http://ufldl.stanford.edu/housenumbers/extra_32x32.mat', 'extra_32x32.mat', 'a93ce644f1a588dc4d68dda5feec44a7']}",
    )
    return argument_parser


def UCF101Config(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser, video (Tensor[T, H, W, C]): the `T` video frames
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "`UCF101 <https://www.crcv.ucf.edu/data/UCF101.php>`_ dataset.\n\nUCF101 is an action recognition video dataset.\nThis dataset consider every video as a collection of video clips of fixed size, specified\nby ``frames_per_clip``, where the step in frames between each clip is given by\n``step_between_clips``.\n\nTo give an example, for 2 videos with 10 and 15 frames respectively, if ``frames_per_clip=5``\nand ``step_between_clips=5``, the dataset size will be (2 + 3) = 5, where the first two\nelements will come from video 1, and the next three elements from video 2.\nNote that we drop clips which do not have exactly ``frames_per_clip`` elements, so not all\nframes in a video might be present.\n\nInternally, it uses a VideoClips object to handle clip creation."
    argument_parser.add_argument(
        "--root",
        type=int,
        help="Root directory of the UCF101 Dataset.",
        required=True,
        default=1,
    )
    argument_parser.add_argument(
        "--annotation_path",
        type=str,
        help="path to the folder containing the split files",
    )
    argument_parser.add_argument(
        "--frames_per_clip",
        type=int,
        help="number of frames in a clip.",
        required=True,
        default=1,
    )
    argument_parser.add_argument(
        "--step_between_clips",
        type=int,
        help="number of frames between each clip.",
        default=1,
    )
    argument_parser.add_argument(
        "--fold",
        type=int,
        help="which fold to use. Should be between 1 and 3.",
        default=1,
    )
    argument_parser.add_argument(
        "--train",
        type=bool,
        help="if ``True``, creates a dataset from the train split,\n        otherwise from the ``test`` split.",
        default=True,
    )
    argument_parser.add_argument(
        "--transform",
        type=int,
        help="A function/transform that  takes in a TxHxWxC video\n        and returns a transformed version.",
        default=0,
    )
    argument_parser.add_argument(
        "--_precomputed_metadata", help=None, required=True, default=0
    )
    argument_parser.add_argument("--frame_rate", help=None)
    argument_parser.add_argument("--num_workers", help=None, required=True, default=0)
    argument_parser.add_argument("--_video_width", help=None, required=True, default=0)
    argument_parser.add_argument("--_video_height", help=None, required=True)
    argument_parser.add_argument("--_video_min_dimension", help=None, required=True)
    argument_parser.add_argument("--_audio_samples", help=None, required=True)
    return argument_parser


def USPSConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = "`USPS <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#usps>`_ Dataset.\nThe data-format is : [label [index:value ]*256 \\n] * num_lines, where ``label`` lies in ``[1, 10]``.\nThe value for each pixel lies in ``[-1, 1]``. Here we transform the ``label`` into ``[0, 9]``\nand make pixel values in ``[0, 255]``."
    argument_parser.add_argument(
        "--root",
        type=bool,
        help="Root directory of dataset to store``USPS`` data files.",
        required=True,
        default=True,
    )
    argument_parser.add_argument(
        "--train",
        type=bool,
        help="If True, creates dataset from ``usps.bz2``,\n        otherwise from ``usps.t.bz2``.",
        required=True,
        default=True,
    )
    argument_parser.add_argument(
        "--transform",
        type=str,
        help="A function/transform that  takes in an PIL image\n        and returns a transformed version. E.g, ``transforms.RandomCrop``",
    )
    argument_parser.add_argument(
        "--target_transform",
        type=bool,
        help="A function/transform that takes in the\n        target and transforms it.",
        default=False,
    )
    argument_parser.add_argument(
        "--download",
        type=bool,
        help="If true, downloads the dataset from the internet and\n        puts it in root directory. If dataset is already downloaded, it is not\n        downloaded again.",
        required=True,
        default=False,
    )
    argument_parser.add_argument(
        "--split_list",
        type=loads,
        help="",
        required=True,
        default="{'train': ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2', 'usps.bz2', 'ec16c51db3855ca6c91edd34d0e9b197'], 'test': ['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.t.bz2', 'usps.t.bz2', '8ea070ee2aca1ac39742fdd1ef5ed118']}",
    )
    return argument_parser


def VOCDetectionConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = (
        "`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset."
    )
    argument_parser.add_argument(
        "--root",
        type=str,
        help="Root directory of the VOC Dataset.",
        required=True,
        default="2012",
    )
    argument_parser.add_argument(
        "--year",
        type=str,
        help="The dataset year, supports years 2007 to 2012.",
        required=True,
        default="2012",
    )
    argument_parser.add_argument(
        "--image_set",
        type=str,
        help="Select the image_set to use, ``train``, ``trainval`` or ``val``",
        required=True,
        default="train",
    )
    argument_parser.add_argument(
        "--download",
        type=bool,
        help="If true, downloads the dataset from the internet and\n        puts it in root directory. If dataset is already downloaded, it is not\n        downloaded again.\n        ",
        required=True,
        default=False,
    )
    argument_parser.add_argument(
        "--transform",
        type=str,
        help="A function/transform that  takes in an PIL image\n        and returns a transformed version. E.g, ``transforms.RandomCrop``",
    )
    argument_parser.add_argument(
        "--target_transform",
        type=str,
        help="A function/transform that takes in the\n        target and transforms it.",
    )
    argument_parser.add_argument(
        "--transforms",
        type=str,
        help="A function/transform that takes input sample and its target as entry\n        and returns a transformed version.",
    )
    return argument_parser


def VOCSegmentationConfig(argument_parser):
    """
    Set CLI arguments

    :param argument_parser: argument parser
    :type argument_parser: ```ArgumentParser```

    :return: argument_parser
    :rtype: ```ArgumentParser```
    """
    argument_parser.description = (
        "`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset."
    )
    argument_parser.add_argument(
        "--root",
        type=str,
        help="Root directory of the VOC Dataset.",
        required=True,
        default="2012",
    )
    argument_parser.add_argument(
        "--year",
        type=str,
        help="The dataset year, supports years 2007 to 2012.",
        required=True,
        default="2012",
    )
    argument_parser.add_argument(
        "--image_set",
        type=str,
        help="Select the image_set to use, ``train``, ``trainval`` or ``val``",
        required=True,
        default="train",
    )
    argument_parser.add_argument(
        "--download",
        type=bool,
        help="If true, downloads the dataset from the internet and\n        puts it in root directory. If dataset is already downloaded, it is not\n        downloaded again.",
        required=True,
        default=False,
    )
    argument_parser.add_argument(
        "--transform",
        type=str,
        help="A function/transform that  takes in an PIL image\n        and returns a transformed version. E.g, ``transforms.RandomCrop``",
    )
    argument_parser.add_argument(
        "--target_transform",
        type=str,
        help="A function/transform that takes in the\n        target and transforms it.",
    )
    argument_parser.add_argument(
        "--transforms",
        type=str,
        help="A function/transform that takes input sample and its target as entry\n        and returns a transformed version.",
    )
    return argument_parser


__all__ = [
    "CIFAR100Config",
    "CIFAR10Config",
    "Caltech101Config",
    "Caltech256Config",
    "CelebAConfig",
    "CityscapesConfig",
    "CocoCaptionsConfig",
    "CocoDetectionConfig",
    "DatasetFolderConfig",
    "EMNISTConfig",
    "FakeDataConfig",
    "FashionMNISTConfig",
    "Flickr30kConfig",
    "Flickr8kConfig",
    "HMDB51Config",
    "ImageFolderConfig",
    "ImageNetConfig",
    "KMNISTConfig",
    "Kinetics400Config",
    "LSUNClassConfig",
    "LSUNConfig",
    "MNISTConfig",
    "OmniglotConfig",
    "PhotoTourConfig",
    "Places365Config",
    "QMNISTConfig",
    "SBDatasetConfig",
    "SBUConfig",
    "SEMEIONConfig",
    "STL10Config",
    "SVHNConfig",
    "UCF101Config",
    "USPSConfig",
    "VOCDetectionConfig",
    "VOCSegmentationConfig",
    "datasets2classes",
    "load_data_from_torchvision_or_ml_prepare",
]
