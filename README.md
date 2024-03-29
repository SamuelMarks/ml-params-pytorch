ml_params_pytorch
=================
![Python version range](https://img.shields.io/badge/python-3.6%20|%203.7%20|%203.8-blue.svg)
[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Linting, testing, and coverage](https://github.com/SamuelMarks/ml-params-pytorch/workflows/Linting,%20testing,%20and%20coverage/badge.svg)](https://github.com/SamuelMarks/ml-params-pytorch/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

[PyTorch](https://pytorch.org) implementation of ml-params CLI API

## Install dependencies

    pip install -r requirements.txt

## Install package

    pip install .

## Usage

After installing as above, follow usage from https://github.com/SamuelMarks/ml-params

## Sibling projects

| Google | Other vendors |
| -------| ------------- |
| [tensorflow](https://github.com/SamuelMarks/ml-params-tensorflow)  | [_pytorch_](https://github.com/SamuelMarks/ml-params-pytorch) |
| [keras](https://github.com/SamuelMarks/ml-params-keras)  | [skorch](https://github.com/SamuelMarks/ml-params-skorch) |
| [flax](https://github.com/SamuelMarks/ml-params-flax) | [sklearn](https://github.com/SamuelMarks/ml-params-sklearn) |
| [trax](https://github.com/SamuelMarks/ml-params-trax) | [xgboost](https://github.com/SamuelMarks/ml-params-xgboost) |
| [jax](https://github.com/SamuelMarks/ml-params-jax) | [cntk](https://github.com/SamuelMarks/ml-params-cntk) |

## Related official projects

  - [ml-prepare](https://github.com/SamuelMarks/ml-prepare)

---

## Development guide

To make the development of _ml-params-pytorch_ type safer and maintain consistency with the other ml-params implementing projects, the [cdd](https://github.com/offscale/cdd-python) was created.

When PyTorch itself changes—i.e., a new major version of PyTorch is released—then run the `sync_properties`, as shown in the module-level docstring here [`ml_params_pytorch/ml_params/type_generators.py`](ml_params_pytorch/ml_params/type_generators.py);

To synchronise all the various other APIs, edit one and it'll translate to the others, but make sure you select which one is the gold-standard.

As an example, using the `class TorchTrainer` methods as truth, this will update the CLI parsers and config classes:

    python -m cdd sync --class 'ml_params_pytorch/ml_params/config.py' \
                       --class-name 'TrainConfig' \
                       --function 'ml_params_pytorch/ml_params/trainer.py' \
                       --function-name 'TorchTrainer.train' \
                       --argparse-function 'ml_params_pytorch/ml_params/cli.py' \
                       --argparse-function-name 'train_parser' \
                       --truth 'function'

    python -m cdd sync --class 'ml_params_pytorch/ml_params/config.py' \
                       --class-name 'LoadDataConfig' \
                       --function 'ml_params_pytorch/ml_params/trainer.py' \
                       --function-name 'TorchTrainer.load_data' \
                       --argparse-function 'ml_params_pytorch/ml_params/cli.py' \
                       --argparse-function-name 'load_data_parser' \
                       --truth 'function'

    python -m cdd sync --class 'ml_params_pytorch/ml_params/config.py' \
                       --class-name 'LoadModelConfig' \
                       --function 'ml_params_pytorch/ml_params/trainer.py' \
                       --function-name 'TorchTrainer.load_model' \
                       --argparse-function 'ml_params_pytorch/ml_params/cli.py' \
                       --argparse-function-name 'load_model_parser' \
                       --truth 'function'

To generate custom config CLI parsers, run this before ^:

    $ for name in 'activations' 'losses' 'optimizer_lr_schedulers' 'optimizers'; do
        rm 'ml_params_pytorch/ml_params/'"$name"'.py';        
        python -m ml_params_pytorch.ml_params.cdd_cli_gen "$name" 2>/dev/null | xargs python -m cdd gen;
      done

To see what this is doing, here it is expanded for datasets:

   $ python -m cdd gen --name-tpl '{name}Config' \
                       --input-mapping 'ml_params_pytorch.ml_params.type_generators.exposed_datasets' \
                       --emit 'argparse' \
                       --output-filename 'ml_params_pytorch/ml_params/datasets.py'

Cleanup the code everywhere, removing unused imports and autolinting/autoformatting:

    $ fd -epy -x autoflake --remove-all-unused-imports -i {} \;
    $ isort --atomic .
    $ python -m black .

---

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
